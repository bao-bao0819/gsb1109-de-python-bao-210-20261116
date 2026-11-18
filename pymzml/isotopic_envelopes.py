#!/usr/bin/env python
# encoding: utf-8

"""
Isotopic envelope assignment and scoring.

Given a list of observed centroided peaks (m/z, intensity tuples), a set of
candidate charge states, an m/z spacing tolerance and rules for missing peaks,
this module enumerates all isotopic peak groups (envelopes) that can be
explained by each charge state, records every observed peak that supports an
envelope, scores the competing whole-set explanations and selects a
non-overlapping combination of envelopes.

Two design decisions are worth noting:

* Mutually exclusive envelopes never claim the same observed peak. When the
  best scoring combination is searched, envelopes sharing a peak are treated as
  alternatives.
* Equal scores never break ties by input order. If several candidate charges
  explain the data equally well, all tied explanations are returned so the
  caller can decide.

References:

    Senko et al. (1995), J Am Soc Mass Spectrom 6, 229-233.

@author pymzml contributors
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
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

from collections import namedtuple

from pymzml.spec import PROTON


NEUTRON_MASS = 1.0033548378


class IsotopeMember(namedtuple(
    "IsotopeMember",
    ["peak_index", "isotope_index", "mz", "intensity", "mass_error_ppm"],
)):
    """
    Contribution of a single observed peak to an :class:`IsotopicEnvelope`.

    Attributes:
        peak_index (int): index of the peak in the (sorted) input peak list.
        isotope_index (int): isotope position relative to the monoisotopic
            peak (0 = M, 1 = M+1, ...).
        mz (float): observed m/z.
        intensity (float): observed intensity.
        mass_error_ppm (float): signed deviation from the theoretical isotope
            position, in ppm.
    """

    __slots__ = ()


class IsotopicEnvelope(namedtuple(
    "IsotopicEnvelope",
    [
        "charge",
        "monoisotopic_mz",
        "peak_indices",
        "missing_indices",
        "members",
        "intensity",
        "mean_mass_error_ppm",
        "score",
    ],
)):
    """
    A single isotopic peak group (envelope) assignment.

    Attributes:
        charge (int): charge state ``z`` the envelope is assigned to.
        monoisotopic_mz (float): inferred m/z of the M (first isotope) peak.
        peak_indices (tuple): indices into the input peak list of every
            observed peak supporting the envelope.
        missing_indices (tuple): isotope indices (relative to the
            monoisotopic peak) that are absent within the envelope span.
        members (tuple): one :class:`IsotopeMember` per observed peak, ordered
            by isotope index.
        intensity (float): summed intensity of the supporting observed peaks.
        mean_mass_error_ppm (float): mean absolute m/z deviation of the
            supporting peaks from the theoretical isotope positions, in ppm.
        score (float): envelope score; higher is better.
    """

    __slots__ = ()

    @property
    def neutral_mass(self):
        """float: neutral (uncharged) monoisotopic mass of the envelope."""
        return (self.monoisotopic_mz - PROTON) * self.charge

    @property
    def n_observed(self):
        """int: number of observed peaks supporting the envelope."""
        return len(self.peak_indices)

    @property
    def n_missing(self):
        """int: number of absent (missing) isotope positions."""
        return len(self.missing_indices)

    @property
    def n_isotopes(self):
        """int: total number of isotope positions, observed plus missing."""
        return self.n_observed + self.n_missing

    def __repr__(self):
        return (
            "IsotopicEnvelope(charge={0}, monoisotopic_mz={1:.4f}, "
            "peaks={2}, missing={3}, score={4:.4f})".format(
                self.charge,
                self.monoisotopic_mz,
                self.peak_indices,
                self.missing_indices,
                self.score,
            )
        )


class EnvelopeExplanation(namedtuple(
    "EnvelopeExplanation",
    ["envelopes", "score", "covered_peak_indices", "n_unassigned_peaks"],
)):
    """
    A whole-set explanation: a non-overlapping combination of envelopes.

    Attributes:
        envelopes (tuple): :class:`IsotopicEnvelope` objects of the
            explanation, ordered by ascending monoisotopic m/z.
        score (float): summed envelope score.
        covered_peak_indices (frozenset): peak indices claimed by any envelope
            of the explanation. Mutually exclusive envelopes never share an
            entry here.
        n_unassigned_peaks (int): observed peaks that no envelope of this
            explanation claims.
    """

    __slots__ = ()

    @property
    def n_envelopes(self):
        """int: number of envelopes in the explanation."""
        return len(self.envelopes)

    def __repr__(self):
        return (
            "EnvelopeExplanation(n_envelopes={0}, score={1:.4f}, "
            "unassigned={2})".format(
                self.n_envelopes, self.score, self.n_unassigned_peaks
            )
        )


class IsotopicEnvelopeFinder(object):
    """
    Enumerate, score and select isotopic peak groups from centroided peaks.

    For a charge state ``z`` consecutive isotopes of one species are spaced by
    ``1.0033548378 / z`` m/z. The finder walks every observed peak as a
    potential monoisotopic (M) peak, follows the expected spacing within
    ``mz_tolerance`` and accepts missing isotope positions according to the
    configured rules. Each valid group is emitted as an
    :class:`IsotopicEnvelope` carrying every contributing observed peak.

    Keyword Arguments:
        charges (iterable): candidate charge states, e.g. ``[2, 3]``.
        mz_tolerance (float): maximum allowed absolute m/z deviation between an
            observed peak and a theoretical isotope position.
        max_missing_peaks (int): maximum number of missing isotope positions
            allowed within an envelope.
        max_consecutive_missing (int): maximum number of consecutive missing
            isotope positions.
        min_isotopes (int): minimum number of observed peaks an envelope must
            contain.
        max_isotopes (int): maximum isotope index relative to the
            monoisotopic peak an envelope may span. Bounds the search for very
            dense peak lists.
        neutron_mass (float): neutron mass used for the isotope spacing.
    """

    def __init__(
        self,
        charges=(2, 3),
        mz_tolerance=0.02,
        max_missing_peaks=1,
        max_consecutive_missing=1,
        min_isotopes=2,
        max_isotopes=12,
        neutron_mass=NEUTRON_MASS,
    ):
        self.charges = tuple(int(charge) for charge in charges)
        if not self.charges:
            raise ValueError("at least one candidate charge is required")
        if any(charge <= 0 for charge in self.charges):
            raise ValueError("charge states must be positive integers")
        if len(set(self.charges)) != len(self.charges):
            raise ValueError("duplicate candidate charges are not allowed")
        if mz_tolerance <= 0:
            raise ValueError("mz_tolerance must be greater than zero")
        if max_missing_peaks < 0 or max_consecutive_missing < 0:
            raise ValueError("missing peak rules must not be negative")
        if min_isotopes < 1:
            raise ValueError("min_isotopes must be at least 1")
        if max_isotopes < min_isotopes:
            raise ValueError(
                "max_isotopes must not be smaller than min_isotopes"
            )

        self.mz_tolerance = float(mz_tolerance)
        self.max_missing_peaks = int(max_missing_peaks)
        self.max_consecutive_missing = int(max_consecutive_missing)
        self.min_isotopes = int(min_isotopes)
        self.max_isotopes = int(max_isotopes)
        self.neutron_mass = float(neutron_mass)

    @property
    def isotope_spacings(self):
        """dict: mapping of charge state to theoretical isotope spacing in m/z."""
        return {
            charge: self.neutron_mass / charge for charge in self.charges
        }

    def find(self, peaks):
        """
        Enumerate every valid isotopic envelope for every candidate charge.

        Arguments:
            peaks (iterable): ``(mz, intensity)`` tuples of centroided peaks.

        Returns:
            list: all valid :class:`IsotopicEnvelope` objects, ordered by
            candidate charge and ascending monoisotopic m/z. Competing charges
            explaining the same peaks are both returned; ties are not resolved
            here.
        """
        peak_list = self._normalize_peaks(peaks)
        envelopes = []
        for charge in self.charges:
            spacing = self.neutron_mass / charge
            for anchor_index in range(len(peak_list)):
                envelopes.extend(
                    self._envelopes_from_anchor(
                        peak_list, anchor_index, charge, spacing
                    )
                )
        return envelopes

    def score_envelope(self, envelope):
        """
        Score a single envelope.

        The score rewards the number of explained isotope positions and the
        completeness of the observed isotope series, and penalises missing
        positions and the mean mass error. Higher scores are better. The score
        is deterministic and depends only on the envelope data, never on input
        ordering.

        Arguments:
            envelope (IsotopicEnvelope): envelope to score.

        Returns:
            float: envelope score.
        """
        completeness = envelope.n_observed / envelope.n_isotopes
        mass_accuracy = 1.0 / (1.0 + envelope.mean_mass_error_ppm)
        return (
            envelope.n_observed
            + completeness
            + mass_accuracy
            - envelope.n_missing
        )

    def select_explanations(self, peaks, envelopes=None):
        """
        Select non-overlapping whole-set explanations of the peaks.

        Envelopes claiming the same observed peak are mutually exclusive. The
        envelope conflict graph is split into independent components and every
        optimal combination per component is kept, so equal-scoring charges are
        all retained rather than broken by input order.

        Arguments:
            peaks (iterable): ``(mz, intensity)`` tuples of centroided peaks.
            envelopes (list): optional precomputed envelopes from
                :meth:`find`. If omitted, :meth:`find` is called.

        Returns:
            list: all best :class:`EnvelopeExplanation` objects. A single
            element means a unique best explanation; several elements denote a
            score tie. An empty envelope list yields one explanation without
            envelopes.
        """
        peak_list = self._normalize_peaks(peaks)
        if envelopes is None:
            envelopes = self.find(peak_list)
        else:
            envelopes = list(envelopes)

        if not envelopes:
            return [
                EnvelopeExplanation(
                    envelopes=(),
                    score=0.0,
                    covered_peak_indices=frozenset(),
                    n_unassigned_peaks=len(peak_list),
                )
            ]

        component_options = [
            self._best_combinations_for_component(component)
            for component in self._conflict_components(envelopes)
        ]

        explanations = []
        for option_combination in _cartesian_product(component_options):
            chosen = [
                envelope
                for option in option_combination
                for envelope in option
            ]
            covered = frozenset(
                peak_index
                for envelope in chosen
                for peak_index in envelope.peak_indices
            )
            chosen.sort(key=lambda envelope: envelope.monoisotopic_mz)
            explanations.append(
                EnvelopeExplanation(
                    envelopes=tuple(chosen),
                    score=sum(envelope.score for envelope in chosen),
                    covered_peak_indices=covered,
                    n_unassigned_peaks=len(peak_list) - len(covered),
                )
            )

        best_score = max(explanation.score for explanation in explanations)
        winners = [
            explanation
            for explanation in explanations
            if explanation.score == best_score
        ]
        winners.sort(
            key=lambda explanation: tuple(
                (envelope.charge, envelope.monoisotopic_mz)
                for envelope in explanation.envelopes
            )
        )
        return winners

    def assign(self, peaks):
        """
        Convenience wrapper returning envelopes and best explanations at once.

        Arguments:
            peaks (iterable): ``(mz, intensity)`` tuples of centroided peaks.

        Returns:
            tuple: ``(envelopes, explanations)`` as returned by
            :meth:`find` and :meth:`select_explanations`.
        """
        peak_list = self._normalize_peaks(peaks)
        envelopes = self.find(peak_list)
        explanations = self.select_explanations(peak_list, envelopes)
        return envelopes, explanations

    def _envelopes_from_anchor(self, peaks, anchor_index, charge, spacing):
        anchor_mz = peaks[anchor_index][0]
        results = []
        seen_signatures = set()

        def record(assignments, missing):
            # Missing positions are internal gaps: a missing isotope at the
            # tail of an envelope, with no observed isotope after it, carries
            # no evidence and is not reported.
            if missing and missing[-1] >= assignments[-1][1]:
                return
            signature = tuple(
                (peak_index, isotope_index)
                for peak_index, isotope_index, _error in assignments
            )
            if signature in seen_signatures:
                return
            seen_signatures.add(signature)

            members = tuple(
                IsotopeMember(
                    peak_index=peak_index,
                    isotope_index=isotope_index,
                    mz=peaks[peak_index][0],
                    intensity=peaks[peak_index][1],
                    mass_error_ppm=mass_error_ppm,
                )
                for peak_index, isotope_index, mass_error_ppm in assignments
            )
            envelope = IsotopicEnvelope(
                charge=charge,
                monoisotopic_mz=anchor_mz,
                peak_indices=tuple(
                    peak_index
                    for peak_index, _isotope, _error in assignments
                ),
                missing_indices=tuple(missing),
                members=members,
                intensity=sum(member.intensity for member in members),
                mean_mass_error_ppm=sum(
                    abs(member.mass_error_ppm) for member in members
                )
                / len(members),
                score=0.0,
            )
            results.append(
                envelope._replace(score=self.score_envelope(envelope))
            )

        def mark_missing(next_index, isotope_index, assignments, missing, gap):
            if len(missing) >= self.max_missing_peaks:
                return
            if gap + 1 > self.max_consecutive_missing:
                return
            missing.append(isotope_index)
            extend(
                next_index,
                isotope_index + 1,
                assignments,
                missing,
                gap + 1,
            )
            missing.pop()

        def extend(search_index, isotope_index, assignments, missing, gap):
            if len(assignments) >= self.min_isotopes:
                record(assignments, missing)
            if isotope_index > self.max_isotopes:
                return
            if search_index >= len(peaks):
                return

            theoretical_mz = anchor_mz + isotope_index * spacing
            lower_bound = theoretical_mz - self.mz_tolerance
            upper_bound = theoretical_mz + self.mz_tolerance

            # Skip interfering peaks that fall below this isotope window.
            window_start = search_index
            while (
                window_start < len(peaks)
                and peaks[window_start][0] < lower_bound
            ):
                window_start += 1

            if (
                window_start >= len(peaks)
                or peaks[window_start][0] > upper_bound
            ):
                # No observed peak in the window: the position is missing.
                mark_missing(
                    window_start,
                    isotope_index,
                    assignments,
                    missing,
                    gap,
                )
                return

            window_end = window_start
            while (
                window_end < len(peaks)
                and peaks[window_end][0] <= upper_bound
            ):
                window_end += 1

            # Within one isotope window the nearest peak to the theoretical
            # position is the deterministic best match. Alternative peaks in
            # the same window compete through the conflict graph rather than
            # through exponentially many envelope copies.
            candidate_index = min(
                range(window_start, window_end),
                key=lambda index: abs(peaks[index][0] - theoretical_mz),
            )
            observed_mz = peaks[candidate_index][0]
            mass_error_ppm = (
                (observed_mz - theoretical_mz) / theoretical_mz * 1e6
            )
            assignments.append(
                (candidate_index, isotope_index, mass_error_ppm)
            )
            extend(
                candidate_index + 1,
                isotope_index + 1,
                assignments,
                missing,
                0,
            )
            assignments.pop()

            # Alternative: leave the position missing; the window peaks stay
            # available for later isotope positions.
            mark_missing(
                window_start,
                isotope_index,
                assignments,
                missing,
                gap,
            )

        extend(anchor_index + 1, 1, [(anchor_index, 0, 0.0)], [], 0)
        return results

    def _conflict_components(self, envelopes):
        adjacency = {index: set() for index in range(len(envelopes))}
        peak_owners = {}
        for index, envelope in enumerate(envelopes):
            for peak_index in envelope.peak_indices:
                owners = peak_owners.setdefault(peak_index, [])
                for other_index in owners:
                    adjacency[index].add(other_index)
                    adjacency[other_index].add(index)
                owners.append(index)

        visited = set()
        components = []
        for start in range(len(envelopes)):
            if start in visited:
                continue
            stack = [start]
            component = []
            visited.add(start)
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in adjacency[node]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        stack.append(neighbour)
            components.append(
                [envelopes[index] for index in sorted(component)]
            )
        return components

    def _best_combinations_for_component(self, component):
        best_score = 0.0
        winners = []

        def search(start_index, chosen, used_peaks, current_score):
            nonlocal best_score, winners
            if current_score > best_score:
                best_score = current_score
                winners = [tuple(chosen)]
            elif current_score == best_score:
                winners.append(tuple(chosen))

            for index in range(start_index, len(component)):
                envelope = component[index]
                envelope_peaks = set(envelope.peak_indices)
                if used_peaks & envelope_peaks:
                    continue
                chosen.append(envelope)
                search(
                    index + 1,
                    chosen,
                    used_peaks | envelope_peaks,
                    current_score + envelope.score,
                )
                chosen.pop()

        search(0, [], frozenset(), 0.0)
        return winners

    @staticmethod
    def _normalize_peaks(peaks):
        peak_list = [
            (float(mz), float(intensity)) for mz, intensity in peaks
        ]
        peak_list.sort(key=lambda peak: peak[0])
        return peak_list


def _cartesian_product(sequences):
    if not sequences:
        return [()]
    product = [()]
    for sequence in sequences:
        product = [
            prefix + (item,)
            for prefix in product
            for item in sequence
        ]
    return product
