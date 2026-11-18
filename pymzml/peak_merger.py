#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to merge neighboring peaks of a spectrum.

Peaks are grouped based on the m/z distance between them, using the same
conventions as the rest of pymzml, i.e. a ppm based error combined with an
absolute tolerance. A group is only accepted if **all** of its members mutually
satisfy the tolerance, so transitive chaining alone (three peaks where each
adjacent pair matches but the first and last do not) is not silently merged.

The grouping is deterministic and independent of the order in which the peaks
are supplied: peaks are always ordered by m/z (and, for ties, by their original
position). If more than one grouping is compatible with the tolerance rules, an
:class:`AmbiguousPeakGroupError` is raised instead of guessing.
"""

from dataclasses import dataclass
from typing import NamedTuple, Tuple


class AmbiguousPeakGroupError(ValueError):
    """
    Raised when the peaks cannot be assigned to a single unambiguous group.

    This happens when peaks are connected through neighboring pairs that each
    satisfy the tolerance but the connected set as a whole violates the
    pairwise tolerance rule, so different legal groupings exist and none can be
    preferred without additional information.

    Attributes:
        peaks (tuple): ``(m/z, intensity)`` peaks of the ambiguous connected set
        ppm (float): ppm tolerance that was applied
        abs_tolerance (float): absolute tolerance that was applied
    """

    def __init__(self, peaks, ppm, abs_tolerance, message=None):
        self.peaks = tuple(peaks)
        self.ppm = ppm
        self.abs_tolerance = abs_tolerance
        if message is None:
            message = (
                "Peaks {0} are connected by neighboring pairs within {1} ppm "
                "and an absolute tolerance of {2}, but not every pair of the "
                "group satisfies the tolerance; the grouping is therefore "
                "ambiguous".format(self.peaks, ppm, abs_tolerance)
            )
        super().__init__(message)


class MergedPeak(NamedTuple):
    """
    A merged peak together with the original peaks it was derived from.

    Attributes:
        mz (float): intensity weighted m/z of the merged peak.
        intensity (float): summed intensity of the contributing peaks.
        contributors (tuple): tuples of the original ``(m/z, intensity)``
            values that were merged into this peak.
    """

    mz: float
    intensity: float
    contributors: Tuple[Tuple[float, float], ...]

    def as_peak(self):
        """
        Return the merged peak as a plain ``(m/z, intensity)`` tuple.

        Returns:
            tuple: merged m/z and summed intensity
        """
        return (self.mz, self.intensity)


def within_tolerance(mz1, mz2, ppm=5.0, abs_tolerance=0.0):
    """
    Check whether two m/z values are within the allowed error.

    The ppm error and the absolute tolerance are combined by taking the larger
    of the two allowed windows, i.e. a pair is accepted if it satisfies either
    rule. The ppm window is evaluated against the mean of the two m/z values.

    Arguments:
        mz1 (float): first m/z value
        mz2 (float): second m/z value

    Keyword Arguments:
        ppm (float): allowed error in parts per million
        abs_tolerance (float): allowed absolute m/z error

    Returns:
        bool: ``True`` if the distance is within the combined tolerance
    """
    reference_mz = (float(mz1) + float(mz2)) / 2.0
    allowed_error = max(abs(ppm) * reference_mz / 1_000_000.0, abs(abs_tolerance))
    return abs(float(mz1) - float(mz2)) <= allowed_error


@dataclass(frozen=True)
class PeakGroup:
    """
    A validated group of peaks that mutually satisfy the tolerance rules.

    Attributes:
        indices (tuple): positions of the members in the original peak list,
            sorted ascending
        mzs (tuple): member m/z values in the same order as ``indices``
        intensities (tuple): member intensities in the same order as ``indices``
        merged_mz (float): intensity weighted m/z of the group
        merged_intensity (float): summed intensity of the group
    """

    indices: Tuple[int, ...]
    mzs: Tuple[float, ...]
    intensities: Tuple[float, ...]
    merged_mz: float
    merged_intensity: float

    @property
    def contributors(self):
        """
        Return the original ``(m/z, intensity)`` peaks contributing to the group.

        Returns:
            tuple: original peak tuples belonging to the group
        """
        return tuple(zip(self.mzs, self.intensities))

    def as_peak(self):
        """
        Return the group as a plain ``(m/z, intensity)`` tuple.

        Returns:
            tuple: merged m/z and summed intensity
        """
        return (self.merged_mz, self.merged_intensity)

    def as_merged_peak(self):
        """
        Return the group as a :class:`MergedPeak`, including its contributors.

        Returns:
            MergedPeak: merged peak with traceable original peaks
        """
        return MergedPeak(
            self.merged_mz, self.merged_intensity, self.contributors
        )


def _normalize_peaks(peaks):
    normalized = []
    for position, peak in enumerate(peaks):
        try:
            mz = float(peak[0])
            intensity = float(peak[1])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "Peak at position {0} cannot be interpreted as an "
                "(m/z, intensity) pair: {1!r}".format(position, peak)
            ) from error
        normalized.append((mz, intensity))
    return normalized


def _all_pairs_within_tolerance(mzs, ppm, abs_tolerance):
    for pos_a in range(len(mzs)):
        for pos_b in range(pos_a + 1, len(mzs)):
            if not within_tolerance(
                mzs[pos_a], mzs[pos_b], ppm=ppm, abs_tolerance=abs_tolerance
            ):
                return False
    return True


def build_peak_groups(peaks, ppm=5.0, abs_tolerance=0.0):
    """
    Build deterministic peak groups from a list of ``(m/z, intensity)`` peaks.

    Peaks are ordered by m/z (using the original position as a tie breaker) so
    that the result does not depend on the input order. Peaks satisfying the
    combined ppm and absolute tolerance are connected, and every connected set
    is only accepted if all of its members pairwise satisfy the tolerance.
    Otherwise an :class:`AmbiguousPeakGroupError` is raised.

    Arguments:
        peaks (sequence): list, tuple or numpy array of ``(m/z, intensity)``
            pairs

    Keyword Arguments:
        ppm (float): allowed error in parts per million
        abs_tolerance (float): allowed absolute m/z error, combined with the
            ppm error by taking the larger allowed window

    Returns:
        list: list of :class:`PeakGroup` objects, ordered by ascending m/z

    Raises:
        AmbiguousPeakGroupError: if a connected set of peaks does not pairwise
            satisfy the tolerance and therefore has no unique grouping
        ValueError: if an entry cannot be interpreted as an m/z intensity pair
    """
    records = _normalize_peaks(peaks)
    peak_count = len(records)
    order = sorted(range(peak_count), key=lambda idx: (records[idx][0], idx))
    sorted_mzs = [records[idx][0] for idx in order]

    neighbors = [set() for _ in range(peak_count)]
    for pos_a in range(peak_count):
        for pos_b in range(pos_a + 1, peak_count):
            if within_tolerance(
                sorted_mzs[pos_a],
                sorted_mzs[pos_b],
                ppm=ppm,
                abs_tolerance=abs_tolerance,
            ):
                neighbors[pos_a].add(pos_b)
                neighbors[pos_b].add(pos_a)

    component_of = [-1] * peak_count
    component_count = 0
    for start in range(peak_count):
        if component_of[start] != -1:
            continue
        component_of[start] = component_count
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if component_of[neighbor] == -1:
                    component_of[neighbor] = component_count
                    stack.append(neighbor)
        component_count += 1

    groups = []
    for component_id in range(component_count):
        sorted_positions = [
            pos for pos in range(peak_count) if component_of[pos] == component_id
        ]
        component_records = [records[order[pos]] for pos in sorted_positions]
        original_indices = tuple(sorted(order[pos] for pos in sorted_positions))
        member_mzs = tuple(record[0] for record in component_records)
        member_intensities = tuple(record[1] for record in component_records)

        if not _all_pairs_within_tolerance(member_mzs, ppm, abs_tolerance):
            offending_peaks = tuple(zip(member_mzs, member_intensities))
            raise AmbiguousPeakGroupError(offending_peaks, ppm, abs_tolerance)

        total_intensity = sum(member_intensities)
        if total_intensity:
            merged_mz = (
                sum(
                    mz * intensity
                    for mz, intensity in zip(member_mzs, member_intensities)
                )
                / total_intensity
            )
        else:
            merged_mz = sum(member_mzs) / len(member_mzs)

        groups.append(
            PeakGroup(
                indices=original_indices,
                mzs=member_mzs,
                intensities=member_intensities,
                merged_mz=merged_mz,
                merged_intensity=total_intensity,
            )
        )
    return groups


def merge_peaks(peaks, ppm=5.0, abs_tolerance=0.0):
    """
    Merge neighboring peaks according to the ppm and absolute tolerance rules.

    The merged m/z is the intensity weighted mean of the group members and the
    merged intensity is their sum. Every returned :class:`MergedPeak` keeps the
    original ``(m/z, intensity)`` pairs as contributors so that the provenance
    of the merged intensity can be traced back.

    Arguments:
        peaks (sequence): list, tuple or numpy array of ``(m/z, intensity)``
            pairs

    Keyword Arguments:
        ppm (float): allowed error in parts per million
        abs_tolerance (float): allowed absolute m/z error, combined with the
            ppm error by taking the larger allowed window

    Returns:
        list: list of :class:`MergedPeak` objects ordered by ascending m/z

    Raises:
        AmbiguousPeakGroupError: if the peaks do not admit a unique grouping
        ValueError: if an entry cannot be interpreted as an m/z intensity pair
    """
    groups = build_peak_groups(peaks, ppm=ppm, abs_tolerance=abs_tolerance)
    return [group.as_merged_peak() for group in groups]


__all__ = [
    "AmbiguousPeakGroupError",
    "MergedPeak",
    "PeakGroup",
    "build_peak_groups",
    "merge_peaks",
    "within_tolerance",
]
