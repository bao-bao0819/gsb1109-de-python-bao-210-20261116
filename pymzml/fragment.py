#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fragment based support comparison for candidate molecules.

Given one measured spectrum and a number of candidate molecules, this module
matches the theoretical fragments of every candidate -- optionally including
allowed neutral losses -- against the measured peaks.  The matching is solved
as a weighted bipartite assignment problem instead of a greedy peak-by-peak
walk, which guarantees three properties:

* a measured peak can contribute at most once to the score of a candidate,
* the order in which candidates or peaks are supplied never influences the
  result, and
* all assignments that reach the same optimum are kept, so an ambiguous peak
  is reported as several equally good explanations instead of being awarded
  to whichever candidate happened to come first.

The public entry point is :func:`match_fragments`.  A reusable object oriented
variant is available as :class:`FragmentMatcher`.  Both return a
:class:`MatchResult` that keeps the matched assignments, the unmatched peaks
and, per candidate, all optimal explanations.

Example:

>>> fragments = {"b2": 251.1503, "y2": 260.1969}
>>> peaks = [(251.1504, 100.0), (260.1970, 80.0), (400.0, 5.0)]
>>> result = match_fragments(peaks, [{"name": "peptide", "fragments": fragments}],
...                          tolerance=20, tolerance_unit="ppm")
>>> result["peptide"].score
2.0
>>> len(result["peptide"].unmatched_peaks)
1
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Mapping, Sequence

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


# ---------------------------------------------------------------------------
# tolerance helpers
# ---------------------------------------------------------------------------

_PPM_UNITS = {"ppm", "p.p.m.", "p.p.m", "relative", "fraction"}
_DALTON_UNITS = {"da", "dalton", "daltons", "th", "m/z", "mz", "absolute"}

_PEAK_USAGE_PER_CANDIDATE = {
    "per_candidate",
    "per-candidate",
    "candidate",
    "independent",
    "shared",
    "share",
    "reuse",
    "per_peak",
}
_PEAK_USAGE_EXCLUSIVE = {
    "exclusive",
    "global",
    "once",
    "one",
    "single",
    "strict",
    "unique",
}


def _normalise_peak_usage(peak_usage: Any) -> str:
    """Return the canonical peak usage rule (``per_candidate``/``exclusive``)."""
    if isinstance(peak_usage, bool):
        return "exclusive" if peak_usage else "per_candidate"
    if peak_usage is None:
        return "per_candidate"
    key = str(peak_usage).strip().lower().replace(" ", "_")
    if key in _PEAK_USAGE_PER_CANDIDATE:
        return "per_candidate"
    if key in _PEAK_USAGE_EXCLUSIVE:
        return "exclusive"
    raise ValueError(
        "Unknown peak usage rule {0!r}; use 'per_candidate' or 'exclusive'.".format(
            peak_usage
        )
    )


def _parse_tolerance(tolerance, unit=None, spectrum=None):
    """
    Normalise a tolerance into ``(value, unit)``.

    ``unit='ppm'`` accepts both ppm counts (``20``) and fractions
    (``20e-6``).  Values below ``1e-3`` are treated as fractions, larger
    values as ppm counts.  ``unit='Da'`` interprets the value as an absolute
    m/z window.
    """
    if tolerance is None:
        if spectrum is not None:
            precision = getattr(spectrum, "measured_precision", None)
            if precision is not None:
                return float(precision), "ppm"
        return 5e-6, "ppm"

    if isinstance(tolerance, Mapping):
        unit = tolerance.get("unit", unit)
        tolerance = tolerance.get("value", tolerance.get("tolerance"))

    if isinstance(tolerance, str):
        text = tolerance.strip().lower().replace(" ", "")
        if text.endswith("ppm"):
            unit, tolerance = "ppm", text[:-3]
        elif text.endswith("da"):
            unit, tolerance = "Da", text[:-2]
        elif text.endswith("th"):
            unit, tolerance = "Da", text[:-2]
        tolerance = float(tolerance)

    if isinstance(tolerance, (tuple, list)):
        if len(tolerance) != 2:
            raise ValueError("Tolerance tuples must be (value, unit).")
        tolerance, unit = tolerance

    value = float(tolerance)
    if value < 0:
        raise ValueError("Tolerance must not be negative.")

    if unit is None:
        unit = "ppm"
    unit = str(unit).strip().lower()
    if unit in _DALTON_UNITS:
        return value, "Da"
    if unit in _PPM_UNITS:
        if value >= 1e-3:
            value = value / 1_000_000.0
        return value, "ppm"
    raise ValueError(
        "Unknown tolerance unit {0!r}; use 'ppm' or 'Da'.".format(unit)
    )


def _within_tolerance(measured, theoretical, tolerance, unit):
    error = measured - theoretical
    if unit == "Da":
        return abs(error) <= tolerance
    return abs(error) <= abs(theoretical) * tolerance


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralLoss:
    """A neutral loss that may be subtracted from a theoretical fragment."""

    label: str
    mass: float
    weight: float = 1.0

    def as_dict(self):
        return {"label": self.label, "mass": self.mass, "weight": self.weight}


@dataclass(frozen=True)
class Ion:
    """A theoretical ion, i.e. a fragment with an optional neutral loss."""

    label: str
    mz: float
    weight: float = 1.0
    fragment_label: str | None = None
    loss_label: str | None = None
    loss_mass: float = 0.0

    def as_dict(self):
        return {
            "label": self.label,
            "mz": self.mz,
            "weight": self.weight,
            "fragment_label": self.fragment_label,
            "loss_label": self.loss_label,
            "loss_mass": self.loss_mass,
        }


@dataclass(frozen=True)
class FragmentCandidate:
    """Normalised candidate molecule.

    Attributes:
        name: Candidate name.
        fragments: Theoretical fragments as ``(label, mz, weight)`` entries.
        neutral_losses: Candidate specific neutral losses.  Global losses are
            merged in by :func:`match_fragments`.
        metadata: Optional free form information carried through the result.
    """

    name: str
    fragments: tuple[tuple[str, float, float], ...]
    neutral_losses: tuple[NeutralLoss, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def ions(self, neutral_losses: Iterable[NeutralLoss] = (), weights=None):
        """Build all ions, i.e. fragments plus every allowed neutral loss."""
        losses = list(self.neutral_losses)
        seen = {(_loss.label, _loss.mass) for _loss in losses}
        for loss in neutral_losses:
            key = (loss.label, loss.mass)
            if key not in seen:
                seen.add(key)
                losses.append(loss)

        ions = []
        for fragment_label, mz, weight in self.fragments:
            ion_weight = _ion_weight(weight, fragment_label, None, weights)
            if ion_weight > 0:
                ions.append(
                    Ion(
                        label=fragment_label,
                        mz=mz,
                        weight=ion_weight,
                        fragment_label=fragment_label,
                    )
                )
            for loss in losses:
                loss_mz = mz - loss.mass
                ion_weight = _ion_weight(
                    weight * loss.weight,
                    fragment_label,
                    loss.label,
                    weights,
                )
                if ion_weight > 0:
                    ions.append(
                        Ion(
                            label="{0}-{1}".format(fragment_label, loss.label),
                            mz=loss_mz,
                            weight=ion_weight,
                            fragment_label=fragment_label,
                            loss_label=loss.label,
                            loss_mass=loss.mass,
                        )
                    )
        return tuple(ions)


@dataclass(frozen=True)
class Peak:
    """A measured peak used by the matcher."""

    index: int | None
    mz: float
    intensity: float = 1.0
    weight: float = 1.0

    def __iter__(self):
        yield self.mz
        yield self.intensity

    def __len__(self):
        return 2

    def __getitem__(self, item):
        return (self.mz, self.intensity)[item]

    def as_tuple(self):
        return (self.mz, self.intensity)

    def as_dict(self):
        return {
            "index": self.index,
            "mz": self.mz,
            "intensity": self.intensity,
            "weight": self.weight,
        }


def _resolve_weights_argument(weights, candidate=None, candidate_index=None):
    """Select the weight specification belonging to one candidate."""
    if weights is None:
        return None
    if isinstance(weights, Mapping):
        if candidate is not None:
            name = getattr(candidate, "name", None)
            if name is not None and name in weights:
                nested = weights[name]
                if isinstance(nested, Mapping) or isinstance(nested, Sequence):
                    if not isinstance(nested, (str, bytes)):
                        return nested
        return weights
    if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        if weights and all(
            isinstance(item, (Mapping, Sequence))
            and not isinstance(item, (str, bytes))
            for item in weights
        ):
            index = candidate_index
            if index is not None and index < len(weights):
                return weights[index]
        return weights
    return weights


def _weights_for_candidate(candidate, weights, candidate_index):
    """Resolve candidate level weights into an ion label mapping."""
    resolved = _resolve_weights_argument(weights, candidate, candidate_index)
    if resolved is None:
        return None
    if isinstance(resolved, (int, float)) and not isinstance(resolved, bool):
        return resolved
    if isinstance(resolved, Mapping):
        return resolved
    if isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes)):
        labels = [fragment[0] for fragment in candidate.fragments]
        mapping = {}
        for index, value in enumerate(resolved):
            if index >= len(labels):
                break
            mapping[labels[index]] = float(value)
        return mapping
    return resolved



def _ion_weight(base_weight, fragment_label, loss_label, weights):
    """Resolve the weight of one ion from a flexible weight specification."""
    if weights is None:
        return float(base_weight)
    if isinstance(weights, (int, float)) and not isinstance(weights, bool):
        return float(weights)
    if isinstance(weights, Mapping):
        if loss_label is not None:
            full_label = "{0}-{1}".format(fragment_label, loss_label)
            if full_label in weights:
                return float(weights[full_label])
        if fragment_label in weights:
            return float(weights[fragment_label])
        for key in ("*", "default", "all"):
            if key in weights:
                return float(weights[key])
        return float(base_weight)
    if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        return float(base_weight)
    return float(base_weight)


def _normalise_losses(neutral_losses) -> tuple[NeutralLoss, ...]:
    if neutral_losses is None:
        return ()
    if isinstance(neutral_losses, NeutralLoss):
        return (neutral_losses,)
    if isinstance(neutral_losses, Mapping):
        if "mass" in neutral_losses and len(neutral_losses) <= 4:
            neutral_losses = [neutral_losses]
        else:
            result = []
            for label, spec in neutral_losses.items():
                if isinstance(spec, Mapping):
                    result.append(
                        NeutralLoss(
                            label=str(spec.get("label", label)),
                            mass=float(spec["mass"]),
                            weight=float(spec.get("weight", 1.0)),
                        )
                    )
                elif isinstance(spec, (tuple, list)):
                    mass = float(spec[0])
                    weight = float(spec[1]) if len(spec) > 1 else 1.0
                    result.append(NeutralLoss(str(label), mass, weight))
                else:
                    result.append(NeutralLoss(str(label), float(spec)))
            return tuple(result)
    if isinstance(neutral_losses, (str, bytes)):
        neutral_losses = [neutral_losses]

    result = []
    for index, item in enumerate(neutral_losses):
        if isinstance(item, NeutralLoss):
            result.append(item)
        elif isinstance(item, Mapping):
            label = item.get("label", item.get("name", "loss{0}".format(index + 1)))
            mass = item.get("mass", item.get("loss", item.get("value")))
            result.append(
                NeutralLoss(
                    label=str(label),
                    mass=float(mass),
                    weight=float(item.get("weight", 1.0)),
                )
            )
        elif isinstance(item, str):
            text = item.strip()
            if ":" in text:
                label, mass = text.split(":", 1)
            elif "=" in text:
                label, mass = text.split("=", 1)
            else:
                label, mass = "loss{0}".format(index + 1), text
            result.append(NeutralLoss(label.strip(), float(mass)))
        elif isinstance(item, (tuple, list)):
            if len(item) >= 2:
                result.append(
                    NeutralLoss(
                        str(item[0]),
                        float(item[1]),
                        float(item[2]) if len(item) > 2 else 1.0,
                    )
                )
            else:
                result.append(
                    NeutralLoss("loss{0}".format(index + 1), float(item[0]))
                )
        else:
            result.append(NeutralLoss("loss{0}".format(index + 1), float(item)))
    return tuple(result)


def _normalise_fragments(fragments):
    """Normalise many accepted fragment input shapes.

    Accepted forms include mappings ``label -> mz`` or ``label -> (mz, weight)``,
    sequences of ``mz``, ``(mz, weight)``, ``(label, mz)`` or mappings with
    ``mz``/``weight``/``label`` keys.
    """
    items = []
    if fragments is None:
        raise ValueError("A candidate needs fragments.")
    if hasattr(fragments, "tolist"):
        fragments = fragments.tolist()

    if isinstance(fragments, Mapping):
        for label, spec in fragments.items():
            if isinstance(spec, Mapping):
                if "mz" not in spec:
                    raise ValueError(
                        "Fragment {0!r} is missing an 'mz' value.".format(label)
                    )
                items.append(
                    (
                        str(spec.get("label", label)),
                        float(spec["mz"]),
                        float(spec.get("weight", 1.0)),
                    )
                )
            elif isinstance(spec, (tuple, list)):
                if len(spec) == 0:
                    raise ValueError(
                        "Fragment {0!r} has an empty specification.".format(label)
                    )
                mass = float(spec[0])
                weight = float(spec[1]) if len(spec) > 1 else 1.0
                items.append((str(label), mass, weight))
            else:
                items.append((str(label), float(spec), 1.0))
    elif isinstance(fragments, Sequence) and not isinstance(
        fragments, (str, bytes)
    ):
        for index, item in enumerate(fragments):
            default_label = "f{0}".format(index + 1)
            if isinstance(item, Mapping):
                if "mz" not in item and "mass" not in item:
                    raise ValueError(
                        "Fragment entry {0} is missing an 'mz' value.".format(index)
                    )
                mass = item.get("mz", item.get("mass"))
                items.append(
                    (
                        str(item.get("label", default_label)),
                        float(mass),
                        float(item.get("weight", 1.0)),
                    )
                )
            elif isinstance(item, (tuple, list)):
                if len(item) == 1:
                    items.append((default_label, float(item[0]), 1.0))
                elif len(item) == 2:
                    first, second = item
                    if isinstance(first, (str, bytes)):
                        items.append((str(first), float(second), 1.0))
                    else:
                        items.append(
                            (default_label, float(first), float(second))
                        )
                else:
                    items.append(
                        (str(item[0]), float(item[1]), float(item[2]))
                    )
            else:
                items.append((default_label, float(item), 1.0))
    else:
        raise TypeError(
            "Fragments must be a mapping or a sequence, got {0}.".format(
                type(fragments).__name__
            )
        )
    return tuple(items)


def _normalise_candidate(candidate, index):
    """Turn any accepted candidate representation into FragmentCandidate."""
    if isinstance(candidate, FragmentCandidate):
        if candidate.name:
            return candidate
        return replace(candidate, name="candidate{0}".format(index + 1))

    metadata = {}
    if isinstance(candidate, Mapping):
        if "fragments" in candidate:
            fragments = candidate["fragments"]
            name = candidate.get(
                "name", candidate.get("id", "candidate{0}".format(index + 1))
            )
            losses = candidate.get("neutral_losses", candidate.get("losses", ()))
            metadata = {
                key: value
                for key, value in candidate.items()
                if key not in ("fragments", "name", "id", "neutral_losses", "losses")
            }
            return FragmentCandidate(
                name=str(name),
                fragments=_normalise_fragments(fragments),
                neutral_losses=_normalise_losses(losses),
                metadata=metadata,
            )
        # A plain mapping of fragment label -> m/z describes one candidate.
        return FragmentCandidate(
            name="candidate{0}".format(index + 1),
            fragments=_normalise_fragments(candidate),
        )

    if isinstance(candidate, (list, tuple)):
        return FragmentCandidate(
            name="candidate{0}".format(index + 1),
            fragments=_normalise_fragments(candidate),
        )

    fragments = getattr(candidate, "fragments", None)
    if fragments is None:
        fragments = getattr(candidate, "peaks", None)
    if fragments is None:
        raise TypeError(
            "Cannot interpret candidate {0!r} as a fragment list.".format(candidate)
        )
    name = getattr(candidate, "name", None) or getattr(
        candidate, "id", "candidate{0}".format(index + 1)
    )
    losses = getattr(candidate, "neutral_losses", ())
    if not losses:
        losses = getattr(candidate, "losses", ())
    return FragmentCandidate(
        name=str(name),
        fragments=_normalise_fragments(fragments),
        neutral_losses=_normalise_losses(losses),
        metadata=getattr(candidate, "metadata", {}),
    )


def _is_numeric_fragment_entry(item):
    if isinstance(item, bool):
        return False
    if isinstance(item, (int, float)):
        return True
    if isinstance(item, Mapping):
        return "mz" in item or "mass" in item
    if isinstance(item, (list, tuple)):
        return len(item) > 0 and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in item
        )
    return False


def _looks_like_fragment_sequence(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(_is_numeric_fragment_entry(item) for item in value)
    )


def _looks_like_candidate(value, allow_sequence=True):
    """Identify a candidate object rather than a bare fragment sequence."""
    if isinstance(value, FragmentCandidate):
        return True
    if getattr(value, "fragments", None) is not None:
        return True
    if getattr(value, "peaks", None) is not None and not isinstance(
        value, (list, tuple)
    ):
        return True
    if isinstance(value, Mapping):
        return "fragments" in value
    if allow_sequence and isinstance(value, (list, tuple)):
        if not value or _looks_like_fragment_sequence(value):
            return False
        return all(_looks_like_candidate(item) for item in value)
    return False


def _normalise_candidates(candidates):
    if isinstance(candidates, FragmentCandidate):
        candidates = [candidates]
    elif isinstance(candidates, Mapping):
        if "fragments" in candidates:
            candidates = [candidates]
        else:
            values = list(candidates.values())
            if values and all(
                _looks_like_candidate(value)
                or _looks_like_fragment_sequence(value)
                for value in values
            ):
                normalised = []
                for index, (name, value) in enumerate(candidates.items()):
                    if _looks_like_fragment_sequence(value):
                        value = list(value)
                    item = _normalise_candidate(value, index)
                    normalised.append(replace(item, name=str(name)))
                return tuple(normalised)
            return (_normalise_candidate(candidates, 0),)
    if isinstance(candidates, (str, bytes)):
        raise TypeError("Candidates must be a sequence or mapping.")
    if isinstance(candidates, Sequence):
        if _looks_like_fragment_sequence(candidates):
            # A bare sequence of fragment m/z values is one candidate.
            return (_normalise_candidate(list(candidates), 0),)
        if all(
            isinstance(candidate, Mapping) and "fragments" not in candidate
            and "mz" not in candidate and "mass" not in candidate
            for candidate in candidates
        ):
            # A sequence of fragment-label mappings describes several
            # candidates, one mapping per candidate.
            return tuple(
                _normalise_candidate(candidate, index)
                for index, candidate in enumerate(candidates)
            )
        if candidates and all(
            _looks_like_candidate(candidate)
            or _looks_like_fragment_sequence(candidate)
            for candidate in candidates
        ):
            return tuple(
                _normalise_candidate(
                    list(candidate)
                    if _looks_like_fragment_sequence(candidate)
                    else candidate,
                    index,
                )
                for index, candidate in enumerate(candidates)
            )
        if candidates and all(
            isinstance(candidate, (list, tuple))
            and all(isinstance(item, (list, tuple)) for item in candidate)
            for candidate in candidates
        ):
            return tuple(
                _normalise_candidate(candidate, index)
                for index, candidate in enumerate(candidates)
            )
        return (_normalise_candidate(list(candidates), 0),)
    raise TypeError(
        "Candidates must be a sequence or mapping, got {0}.".format(
            type(candidates).__name__
        )
    )


def _normalise_peaks(spectrum, peak_type="centroided"):
    """Return ``(peaks, measured_precision)`` for many peak input shapes."""
    precision = getattr(spectrum, "measured_precision", None)
    peaks = None
    if hasattr(spectrum, "peaks") and callable(getattr(spectrum, "peaks")):
        try:
            peaks = spectrum.peaks(peak_type)
        except (KeyError, TypeError):
            peaks = None
        if peaks is None or len(peaks) == 0:
            for fallback in ("centroided", "raw"):
                try:
                    fallback_peaks = spectrum.peaks(fallback)
                except (KeyError, TypeError):
                    continue
                if fallback_peaks is not None and len(fallback_peaks) != 0:
                    peaks = fallback_peaks
                    break
    if peaks is None:
        peaks = spectrum

    if hasattr(peaks, "tolist"):
        peaks = peaks.tolist()
    if isinstance(peaks, Mapping):
        peaks = list(peaks.items())

    normalised = []
    for index, peak in enumerate(peaks):
        if isinstance(peak, Peak):
            normalised.append(
                Peak(peak.index if peak.index is not None else index, peak.mz,
                     peak.intensity, peak.weight)
            )
        elif isinstance(peak, Mapping):
            mass = peak.get("mz", peak.get("mass"))
            intensity = peak.get("i", peak.get("intensity", 1.0))
            weight = peak.get("weight", 1.0)
            normalised.append(
                Peak(index, float(mass), float(intensity), float(weight))
            )
        elif isinstance(peak, (tuple, list)):
            if len(peak) == 0:
                raise ValueError("Empty peak at index {0}.".format(index))
            mass = float(peak[0])
            intensity = float(peak[1]) if len(peak) > 1 else 1.0
            weight = float(peak[2]) if len(peak) > 2 else 1.0
            normalised.append(Peak(index, mass, intensity, weight))
        else:
            normalised.append(Peak(index, float(peak), 1.0, 1.0))
    return tuple(normalised), precision

def _normalise_intensity_weighting(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "linear" if value else "none"
    if isinstance(value, (int, float)):
        if value == 0:
            return "none"
        if value == 1:
            return "linear"
        return float(value)
    key = str(value).strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "linear": "linear",
        "intensity": "linear",
        "relative": "linear",
        "sqrt": "sqrt",
        "square_root": "sqrt",
        "log": "log",
        "logarithmic": "log",
    }
    if key in aliases:
        return aliases[key]
    raise ValueError(
        "Unknown intensity weighting {0!r}; use 'none', 'linear', 'sqrt' or "
        "'log'.".format(value)
    )


def _intensity_factors(peaks, weighting):
    if weighting == "none" or not peaks:
        return [1.0] * len(peaks)
    intensities = [max(float(peak.intensity), 0.0) for peak in peaks]
    maximum = max(intensities) if intensities else 0.0
    if maximum <= 0:
        return [1.0] * len(peaks)
    if weighting == "linear":
        return [intensity / maximum for intensity in intensities]
    if weighting == "sqrt":
        return [math.sqrt(intensity / maximum) for intensity in intensities]
    if weighting == "log":
        denominator = math.log1p(maximum)
        if denominator <= 0:
            return [1.0] * len(peaks)
        return [math.log1p(intensity) / denominator for intensity in intensities]
    return [
        intensity / maximum if maximum > 0 else 1.0 for intensity in intensities
    ]


# ---------------------------------------------------------------------------
# matching machinery
# ---------------------------------------------------------------------------

_NEGATIVE_INFINITY = float("-inf")


def _maximum_weight_matching(edges, n_left, n_right):
    """One maximum weight bipartite matching (min cost flow fallback).

    ``edges`` contains ``(left, right, weight)`` triples.  This routine is only
    used for connected components that are too large for exhaustive dynamic
    programming; it returns one optimal matching without enumerating ties.
    """
    if not edges:
        return ()
    source = n_left + n_right
    sink = source + 1
    graph = [[] for _ in range(sink + 1)]

    def add_edge(start, end, capacity, cost):
        graph[start].append([end, capacity, cost, len(graph[end])])
        graph[end].append([start, 0, -cost, len(graph[start]) - 1])

    for left in range(n_left):
        add_edge(source, left, 1, 0.0)
    for right in range(n_right):
        add_edge(n_left + right, sink, 1, 0.0)
    for left, right, weight in edges:
        add_edge(left, n_left + right, 1, -float(weight))

    while True:
        distance = [float("inf")] * (sink + 1)
        in_queue = [False] * (sink + 1)
        previous = [None] * (sink + 1)
        distance[source] = 0.0
        queue = [source]
        in_queue[source] = True
        while queue:
            node = queue.pop(0)
            in_queue[node] = False
            for edge_index, (target, capacity, cost, _reverse) in enumerate(
                graph[node]
            ):
                if capacity > 0 and distance[node] + cost < distance[target] - 1e-12:
                    distance[target] = distance[node] + cost
                    previous[target] = (node, edge_index)
                    if not in_queue[target]:
                        queue.append(target)
                        in_queue[target] = True
        if previous[sink] is None or distance[sink] >= -1e-12:
            break
        node = sink
        while node != source:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            edge[1] -= 1
            graph[node][edge[3]][1] += 1
            node = parent

    matching = []
    for left in range(n_left):
        for target, capacity, cost, _reverse in graph[left]:
            if n_left <= target < n_left + n_right and capacity == 0 and cost < 0:
                matching.append((left, target - n_left, -cost))
    return tuple(sorted(matching))


@dataclass(frozen=True)
class _ComponentSolution:
    assignments: tuple[tuple[int, int], ...]
    score: float
    truncated: bool = False


def _enumerate_component_solutions(
    peak_ids,
    node_ids,
    edge_map,
    max_explanations,
    state_limit=2_000_000,
):
    """
    Enumerate all maximum weight matchings of one connected component.

    ``edge_map`` maps ``(peak_id, node_id)`` to a positive weight.  A dynamic
    program over the smaller bipartition side is used, which makes it easy to
    recover *all* optimal assignments.  If a component is too large the flow
    fallback returns a single optimal assignment.
    """
    n_peaks = len(peak_ids)
    n_nodes = len(node_ids)
    # The dynamic program walks one side and uses a bit mask over the other
    # side, so the masked side must be the smaller bipartition.
    if n_peaks <= n_nodes:
        left_ids, right_ids = node_ids, peak_ids
        left_is_peak = False
    else:
        left_ids, right_ids = peak_ids, node_ids
        left_is_peak = True

    left_position = {value: index for index, value in enumerate(left_ids)}
    right_position = {value: index for index, value in enumerate(right_ids)}
    adjacency = [[] for _ in left_ids]
    if left_is_peak:
        for (peak_id, node_id), weight in edge_map.items():
            if peak_id in left_position and node_id in right_position:
                adjacency[left_position[peak_id]].append(
                    (right_position[node_id], weight)
                )
    else:
        for (peak_id, node_id), weight in edge_map.items():
            if node_id in left_position and peak_id in right_position:
                adjacency[left_position[node_id]].append(
                    (right_position[peak_id], weight)
                )
    for row in adjacency:
        row.sort(key=lambda item: (item[0], -item[1]))

    n_right = len(right_ids)
    if n_right > 20:
        return _component_solution_fallback(
            adjacency, left_ids, right_ids, left_is_peak, max_explanations
        )

    dp = [{0: 0.0}]
    for row in adjacency:
        previous = dp[-1]
        current = dict(previous)
        for mask, score in previous.items():
            for right, weight in row:
                bit = 1 << right
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + weight
                if new_score > current.get(new_mask, _NEGATIVE_INFINITY) + 1e-12:
                    current[new_mask] = new_score
        dp.append(current)
        if len(current) > state_limit:
            return _component_solution_fallback(
                adjacency, left_ids, right_ids, left_is_peak, max_explanations
            )

    best = max(dp[-1].values()) if dp[-1] else 0.0
    if best <= 0:
        return _ComponentSolution((), 0.0, False)

    limit = max_explanations
    if limit is None:
        limit = 10**9
    else:
        limit = int(limit)

    solutions = []
    truncated = False

    def reconstruct(index, mask, chosen):
        nonlocal truncated
        if index == 0:
            solutions.append(tuple(sorted(chosen)))
            return
        if len(solutions) >= limit:
            truncated = True
            return
        target = dp[index][mask]
        if abs(dp[index - 1].get(mask, _NEGATIVE_INFINITY) - target) <= 1e-12:
            reconstruct(index - 1, mask, chosen)
        for right, weight in adjacency[index - 1]:
            bit = 1 << right
            if not mask & bit:
                continue
            previous_mask = mask ^ bit
            previous_score = dp[index - 1].get(
                previous_mask, _NEGATIVE_INFINITY
            )
            if abs(previous_score + weight - target) <= 1e-12:
                left = left_ids[index - 1]
                if left_is_peak:
                    pair = (left, right_ids[right])
                else:
                    pair = (right_ids[right], left)
                reconstruct(index - 1, previous_mask, chosen + [pair])

    for mask, score in sorted(dp[-1].items()):
        if abs(score - best) > 1e-12:
            continue
        reconstruct(len(left_ids), mask, [])
        if len(solutions) > limit:
            break
    # Different branches can, in rare symmetric cases, yield the same matching.
    unique = []
    seen = set()
    for solution in solutions:
        key = tuple(sorted(solution))
        if key not in seen:
            seen.add(key)
            unique.append(tuple(sorted(solution)))
    unique.sort()
    return _ComponentSolution(tuple(unique), best, truncated)


def _component_solution_fallback(
    adjacency, left_ids, right_ids, left_is_peak, max_explanations
):
    edges = []
    for left_index, row in enumerate(adjacency):
        for right_index, weight in row:
            if left_is_peak:
                edges.append((left_index, right_index, weight))
            else:
                edges.append((right_index, left_index, weight))
    if left_is_peak:
        n_left, n_right = len(left_ids), len(right_ids)
    else:
        n_left, n_right = len(right_ids), len(left_ids)

    def to_pairs(matching):
        pairs = []
        for left_index, right_index, weight in matching:
            if left_is_peak:
                peak_id = left_ids[left_index]
                node_id = right_ids[right_index]
            else:
                peak_id = right_ids[left_index]
                node_id = left_ids[right_index]
            pairs.append((peak_id, node_id))
        return tuple(sorted(pairs))

    best_matching = _maximum_weight_matching(edges, n_left, n_right)
    best_score = sum(weight for _left, _right, weight in best_matching)
    limit = max_explanations if max_explanations is not None else 10**9
    solutions = {to_pairs(best_matching)}

    # Discover alternative optima by forbidding one edge of an already known
    # optimum at a time.  This does not enumerate every tie, but it keeps the
    # genuinely ambiguous alternatives that matter for reporting.
    queue = [best_matching]
    seen_matchings = {tuple(sorted(best_matching))}
    while queue and len(solutions) < limit:
        current = queue.pop()
        for forbidden in current:
            if len(solutions) >= limit:
                break
            remaining = [edge for edge in edges if edge != forbidden]
            alternative = _maximum_weight_matching(remaining, n_left, n_right)
            alternative_score = sum(
                weight for _left, _right, weight in alternative
            )
            if abs(alternative_score - best_score) > 1e-9:
                continue
            key = tuple(sorted(alternative))
            if key in seen_matchings:
                continue
            seen_matchings.add(key)
            solutions.add(to_pairs(alternative))
            queue.append(alternative)

    ordered = tuple(sorted(solutions))
    return _ComponentSolution(ordered, best_score, len(ordered) >= limit)


def _components(peak_ids, node_ids, edge_map):
    """Split a sparse bipartite graph into connected components."""
    peak_adjacency = defaultdict(list)
    node_adjacency = defaultdict(list)
    for peak_id, node_id in edge_map:
        peak_adjacency[peak_id].append(node_id)
        node_adjacency[node_id].append(peak_id)

    visited_peaks = set()
    visited_nodes = set()
    components = []
    for start in peak_ids:
        if start in visited_peaks or not peak_adjacency[start]:
            continue
        peaks = []
        nodes = []
        stack = [("peak", start)]
        while stack:
            kind, value = stack.pop()
            if kind == "peak":
                if value in visited_peaks:
                    continue
                visited_peaks.add(value)
                peaks.append(value)
                for neighbour in peak_adjacency[value]:
                    if neighbour not in visited_nodes:
                        stack.append(("node", neighbour))
            else:
                if value in visited_nodes:
                    continue
                visited_nodes.add(value)
                nodes.append(value)
                for neighbour in node_adjacency[value]:
                    if neighbour not in visited_peaks:
                        stack.append(("peak", neighbour))
        components.append((tuple(sorted(peaks)), tuple(sorted(nodes))))
    return components


def _enumerate_matchings(peak_ids, node_ids, edges, max_explanations):
    """Enumerate all maximum weight matchings of the whole bipartite graph."""
    edge_map = {}
    for peak_id, node_id, weight in edges:
        if weight <= 0:
            continue
        key = (peak_id, node_id)
        if weight > edge_map.get(key, _NEGATIVE_INFINITY):
            edge_map[key] = weight

    combinations = [((), 0.0, False)]
    truncated = False
    for component_peaks, component_nodes in _components(
        peak_ids, node_ids, edge_map
    ):
        component_edges = {
            key: weight
            for key, weight in edge_map.items()
            if key[0] in component_peaks and key[1] in component_nodes
        }
        solution = _enumerate_component_solutions(
            component_peaks, component_nodes, component_edges, max_explanations
        )
        truncated = truncated or solution.truncated
        merged = []
        for previous_assignment, previous_score, previous_truncated in combinations:
            for component_assignment in solution.assignments:
                merged.append(
                    (
                        tuple(sorted(previous_assignment + component_assignment)),
                        previous_score + solution.score,
                        previous_truncated or truncated,
                    )
                )
                if len(merged) >= max_explanations:
                    merged = merged[:max_explanations]
                    truncated = True
                    break
            if len(merged) >= max_explanations:
                break
        combinations = merged
        if not combinations:
            combinations = [((), 0.0, truncated)]

    # Keep only genuinely optimal explanations.
    if not combinations:
        return (), 0.0, truncated
    best = max(score for _assignment, score, _flag in combinations)
    optimal = [
        (assignment, score, flag)
        for assignment, score, flag in combinations
        if abs(score - best) <= 1e-9
    ]
    optimal.sort(key=lambda item: item[0])
    return tuple(optimal), best, truncated


# ---------------------------------------------------------------------------
# result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Assignment:
    """One measured peak matched to one theoretical ion."""

    peak_index: int
    peak_mz: float
    peak_intensity: float
    ion_label: str
    ion_mz: float
    error: float
    error_ppm: float
    weight: float
    score: float

    def as_dict(self):
        return {
            "peak_index": self.peak_index,
            "peak_mz": self.peak_mz,
            "peak_intensity": self.peak_intensity,
            "ion_label": self.ion_label,
            "ion_mz": self.ion_mz,
            "error": self.error,
            "error_ppm": self.error_ppm,
            "weight": self.weight,
            "score": self.score,
        }


@dataclass(frozen=True)
class CandidateMatch:
    """Matching result for a single candidate."""

    name: str
    index: int
    score: float
    matched_weight: float
    total_weight: float
    coverage: float
    support: float
    assignments: tuple[Assignment, ...]
    explanations: tuple[tuple[Assignment, ...], ...]
    unmatched_peaks: tuple[Peak, ...]
    matched_peak_indices: tuple[int, ...]
    unmatched_ion_labels: tuple[str, ...]
    ion_count: int
    matched_ion_count: int
    explanations_truncated: bool = False
    candidate: FragmentCandidate | None = field(default=None, repr=False)

    @property
    def n_explanations(self):
        return len(self.explanations)

    @property
    def ambiguous(self):
        return len(self.explanations) > 1

    @property
    def unmatched_peak_indices(self):
        return tuple(peak.index for peak in self.unmatched_peaks)

    @property
    def matched_peak_count(self):
        return len(self.assignments)

    @property
    def matched_ion_labels(self):
        return tuple(assignment.ion_label for assignment in self.assignments)

    @property
    def alternative_assignments(self):
        return self.explanations

    def __getitem__(self, key):
        if isinstance(key, str) and (
            key in self.__dataclass_fields__ or hasattr(type(self), key)
        ):
            return getattr(self, key)
        raise KeyError(key)

    def as_dict(self):
        return {
            "name": self.name,
            "index": self.index,
            "score": self.score,
            "matched_weight": self.matched_weight,
            "total_weight": self.total_weight,
            "coverage": self.coverage,
            "support": self.support,
            "assignments": [item.as_dict() for item in self.assignments],
            "explanations": [
                [item.as_dict() for item in explanation]
                for explanation in self.explanations
            ],
            "n_explanations": self.n_explanations,
            "ambiguous": self.ambiguous,
            "unmatched_peaks": [peak.as_dict() for peak in self.unmatched_peaks],
            "matched_peak_indices": list(self.matched_peak_indices),
            "unmatched_ions": list(self.unmatched_ion_labels),
            "ion_count": self.ion_count,
            "matched_ion_count": self.matched_ion_count,
            "explanations_truncated": self.explanations_truncated,
        }


@dataclass(frozen=True)
class GlobalAssignment:
    """One peak assignment in a globally exclusive explanation."""

    candidate_index: int
    candidate_name: str
    assignment: Assignment

    def as_dict(self):
        return {
            "candidate_index": self.candidate_index,
            "candidate_name": self.candidate_name,
            "assignment": self.assignment.as_dict(),
        }


@dataclass(frozen=True)
class GlobalExplanation:
    """A complete, globally optimal use of the measured peaks."""

    assignments: tuple[GlobalAssignment, ...]
    score: float
    truncated: bool = False

    @property
    def by_peak(self):
        return {item.assignment.peak_index: item for item in self.assignments}

    def as_dict(self):
        return {
            "assignments": [item.as_dict() for item in self.assignments],
            "score": self.score,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class MatchResult(Mapping):
    """Full result of a fragment matching run.

    The object behaves like a mapping from candidate name to
    :class:`CandidateMatch`, while also exposing the complete peak list, the
    globally unmatched peaks and (for the exclusive rule) all globally optimal
    explanations.
    """

    candidates: tuple[CandidateMatch, ...]
    peaks: tuple[Peak, ...]
    unmatched_peaks: tuple[Peak, ...]
    peak_usage: str
    tolerance: float
    tolerance_unit: str
    global_explanations: tuple[GlobalExplanation, ...] = ()
    explanations_truncated: bool = False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.candidates[key]
        if isinstance(key, str):
            for candidate in self.candidates:
                if candidate.name == key:
                    return candidate
            if key in self.__dataclass_fields__ or key in (
                "scores",
                "names",
                "best",
                "best_score",
                "best_candidates",
                "ranking",
                "n_global_explanations",
            ):
                return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for candidate in self.candidates:
            yield candidate.name

    def __len__(self):
        return len(self.candidates)

    @property
    def names(self):
        return tuple(candidate.name for candidate in self.candidates)

    @property
    def scores(self):
        return {candidate.name: candidate.score for candidate in self.candidates}

    @property
    def best_score(self):
        """Highest score over all candidates (``0.0`` when there are none)."""
        if not self.candidates:
            return 0.0
        return max(candidate.score for candidate in self.candidates)

    @property
    def ranking(self):
        """Candidates sorted by score, ties keep the input order."""
        return tuple(
            sorted(self.candidates, key=lambda candidate: -candidate.score)
        )

    @property
    def best_candidates(self):
        """All candidates sharing the highest score (never an arbitrary one)."""
        if not self.candidates:
            return ()
        best = max(candidate.score for candidate in self.candidates)
        return tuple(
            candidate
            for candidate in self.candidates
            if abs(candidate.score - best) <= 1e-9
        )

    @property
    def best(self):
        """First best candidate, or ``None`` for an empty result."""
        best = self.best_candidates
        return best[0] if best else None

    @property
    def n_global_explanations(self):
        return len(self.global_explanations)

    def as_dict(self):
        return {
            "peak_usage": self.peak_usage,
            "tolerance": self.tolerance,
            "tolerance_unit": self.tolerance_unit,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "scores": self.scores,
            "unmatched_peaks": [peak.as_dict() for peak in self.unmatched_peaks],
            "global_explanations": [
                explanation.as_dict()
                for explanation in self.global_explanations
            ],
            "explanations_truncated": self.explanations_truncated,
        }


FragmentMatchResult = MatchResult


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_OPTION_ALIASES = {
    "mass_tolerance": "tolerance",
    "mz_tolerance": "tolerance",
    "mass_precision": "tolerance",
    "precision": "tolerance",
    "unit": "tolerance_unit",
    "mass_tolerance_unit": "tolerance_unit",
    "peak_rule": "peak_usage",
    "peak_usage_rule": "peak_usage",
    "usage": "peak_usage",
    "ion_weights": "weights",
    "fragment_weights": "weights",
    "losses": "neutral_losses",
    "allowed_losses": "neutral_losses",
    "neutral_loss": "neutral_losses",
    "intensity_weight": "intensity_weighting",
    "intensity_weighting_mode": "intensity_weighting",
    "max_alternatives": "max_explanations",
    "max_solutions": "max_explanations",
}


def _resolve_options(options):
    resolved = {}
    for key, value in options.items():
        canonical = _OPTION_ALIASES.get(key, key)
        if canonical in resolved:
            raise TypeError(
                "Option {0!r} was supplied more than once.".format(canonical)
            )
        resolved[canonical] = value
    return resolved


def match_fragments(
    spectrum,
    candidates,
    neutral_losses=None,
    *,
    tolerance=None,
    tolerance_unit=None,
    weights=None,
    intensity_weighting="none",
    peak_usage="per_candidate",
    max_explanations=100,
    spectrum_peak_type="centroided",
    min_intensity=None,
    **options,
):
    """
    Compare candidate molecules against a measured spectrum.

    Arguments:
        spectrum: Measured spectrum.  May be a :class:`pymzml.spec.Spectrum`,
            a sequence of ``(m/z, intensity)`` peaks or a sequence of m/z
            values.
        candidates: One or more candidates.  Each candidate may be a
            :class:`FragmentCandidate`, a mapping such as
            ``{"name": "A", "fragments": {...}, "neutral_losses": [...]}``,
            a plain ``label -> m/z`` mapping or a sequence of fragment m/z
            values.
        neutral_losses: Optional neutral losses applied to every candidate.
            Losses are subtracted from the fragment m/z, so a loss of
            ``18.0106`` models water loss.  Entries may be numbers, labels,
            ``(label, mass)`` tuples, ``"H2O:18.0106"`` strings or mappings
            with ``mass``/``weight`` keys.

    Keyword Arguments:
        tolerance: Mass tolerance.  With the default unit ``ppm`` both
            ``20`` and ``20e-6`` are accepted.  ``None`` uses the spectrum's
            ``measured_precision`` when available, otherwise ``5e-6``.
        tolerance_unit: ``"ppm"`` (default) or ``"Da"``.
        weights: Ion weights.  May be a number, a mapping from ion/fragment
            label to weight or a sequence aligned with the candidate's
            fragments.  For losses the fragment weight is multiplied by the
            loss weight unless an exact ``fragment-loss`` label is given.
        intensity_weighting: ``"none"`` (default), ``"linear"``, ``"sqrt"``
            or ``"log"``.  Controls how peak intensity scales an edge weight.
        peak_usage: ``"per_candidate"`` (default) allows a peak to support
            several candidates, but never more than once within a candidate.
            ``"exclusive"`` assigns each peak to at most one candidate and
            solves all candidates jointly, so a peak cannot be captured by
            whichever candidate appears first.
        max_explanations: Maximum number of equally optimal explanations kept
            per candidate/global solution.  ``None`` means unlimited.
        spectrum_peak_type: Peak type requested from a spectrum object.
        min_intensity: Ignore measured peaks below this intensity.

    Returns:
        MatchResult: Scores, assignments, unmatched peaks and every optimal
        explanation.
    """
    options = _resolve_options(options)
    if "neutral_losses" in options:
        if neutral_losses is not None:
            raise TypeError("neutral_losses was supplied more than once.")
        neutral_losses = options.pop("neutral_losses")
    for key in (
        "tolerance",
        "tolerance_unit",
        "weights",
        "intensity_weighting",
        "peak_usage",
        "max_explanations",
        "spectrum_peak_type",
        "min_intensity",
    ):
        if key in options:
            value = options.pop(key)
            if key == "tolerance":
                tolerance = value
            elif key == "tolerance_unit":
                tolerance_unit = value
            elif key == "weights":
                weights = value
            elif key == "intensity_weighting":
                intensity_weighting = value
            elif key == "peak_usage":
                peak_usage = value
            elif key == "max_explanations":
                max_explanations = value
            elif key == "spectrum_peak_type":
                spectrum_peak_type = value
            elif key == "min_intensity":
                min_intensity = value
    if options:
        raise TypeError(
            "Unexpected keyword argument(s): {0}".format(
                ", ".join(sorted(options))
            )
        )

    if max_explanations is not None and int(max_explanations) < 1:
        raise ValueError("max_explanations must be at least 1 or None.")
    if max_explanations is not None:
        max_explanations = int(max_explanations)

    peaks, measured_precision = _normalise_peaks(spectrum, spectrum_peak_type)
    if tolerance is None and measured_precision is not None:
        tolerance = float(measured_precision)
    tolerance, tolerance_unit = _parse_tolerance(
        tolerance, tolerance_unit, None
    )

    if min_intensity is not None:
        peaks = tuple(peak for peak in peaks if peak.intensity >= min_intensity)
        peaks = tuple(
            Peak(index, peak.mz, peak.intensity, peak.weight)
            for index, peak in enumerate(peaks)
        )

    candidate_objects = _normalise_candidates(candidates)
    global_losses = _normalise_losses(neutral_losses)
    usage = _normalise_peak_usage(peak_usage)
    weighting = _normalise_intensity_weighting(intensity_weighting)
    intensity_factors = _intensity_factors(peaks, weighting)

    ion_sets = []
    for candidate_index, candidate in enumerate(candidate_objects):
        candidate_weights = _weights_for_candidate(
            candidate, weights, candidate_index
        )
        ion_sets.append(candidate.ions(global_losses, candidate_weights))

    if usage == "per_candidate":
        candidate_matches = []
        truncated = False
        for candidate_index, (candidate, ions) in enumerate(
            zip(candidate_objects, ion_sets)
        ):
            match, was_truncated = _match_one_candidate(
                candidate,
                candidate_index,
                ions,
                peaks,
                intensity_factors,
                tolerance,
                tolerance_unit,
                max_explanations,
            )
            candidate_matches.append(match)
            truncated = truncated or was_truncated
        used_peak_indices = set()
        for match in candidate_matches:
            used_peak_indices.update(match.matched_peak_indices)
        unmatched = tuple(
            peak for peak in peaks if peak.index not in used_peak_indices
        )
        return MatchResult(
            candidates=tuple(candidate_matches),
            peaks=peaks,
            unmatched_peaks=unmatched,
            peak_usage=usage,
            tolerance=tolerance,
            tolerance_unit=tolerance_unit,
            global_explanations=(),
            explanations_truncated=truncated,
        )

    return _match_exclusive(
        candidate_objects,
        ion_sets,
        peaks,
        intensity_factors,
        tolerance,
        tolerance_unit,
        max_explanations,
    )


def _peak_edge_weight(peak, ion, intensity_factor):
    return ion.weight * peak.weight * intensity_factor


def _build_assignments(matching, peaks, ions, intensity_factors):
    assignments = []
    for peak_index, ion_index in matching:
        peak = peaks[peak_index]
        ion = ions[ion_index]
        error = peak.mz - ion.mz
        assignments.append(
            Assignment(
                peak_index=peak.index,
                peak_mz=peak.mz,
                peak_intensity=peak.intensity,
                ion_label=ion.label,
                ion_mz=ion.mz,
                error=error,
                error_ppm=(error / ion.mz * 1_000_000.0) if ion.mz else 0.0,
                weight=ion.weight,
                score=_peak_edge_weight(
                    peak, ion, intensity_factors[peak_index]
                ),
            )
        )
    assignments.sort(key=lambda item: (item.peak_index, item.ion_label))
    return tuple(assignments)


def _match_one_candidate(
    candidate,
    candidate_index,
    ions,
    peaks,
    intensity_factors,
    tolerance,
    tolerance_unit,
    max_explanations,
):
    edges = []
    for peak in peaks:
        for ion_index, ion in enumerate(ions):
            if _within_tolerance(peak.mz, ion.mz, tolerance, tolerance_unit):
                weight = _peak_edge_weight(
                    peak, ion, intensity_factors[peak.index]
                )
                if weight > 0:
                    edges.append((peak.index, ion_index, weight))

    peak_ids = tuple(peak.index for peak in peaks)
    node_ids = tuple(range(len(ions)))
    explanations, best_score, truncated = _enumerate_matchings(
        peak_ids,
        node_ids,
        edges,
        max_explanations if max_explanations is not None else 10**9,
    )

    assignment_explanations = tuple(
        _build_assignments(
            matching,
            peaks,
            ions,
            intensity_factors,
        )
        for matching, _score, _flag in explanations
    )
    if not assignment_explanations:
        assignment_explanations = ((),)
    primary = assignment_explanations[0]
    matched_peak_indices = tuple(
        sorted({assignment.peak_index for assignment in primary})
    )
    matched_peak_set = set(matched_peak_indices)
    unmatched_peaks = tuple(
        peak for peak in peaks if peak.index not in matched_peak_set
    )
    matched_ions = {assignment.ion_label for assignment in primary}
    total_weight = sum(ion.weight for ion in ions)
    matched_weight = sum(assignment.weight for assignment in primary)
    return (
        CandidateMatch(
            name=candidate.name,
            index=candidate_index,
            score=best_score,
            matched_weight=matched_weight,
            total_weight=total_weight,
            coverage=(len(matched_ions) / len(ions)) if ions else 0.0,
            support=(matched_weight / total_weight) if total_weight else 0.0,
            assignments=primary,
            explanations=assignment_explanations,
            unmatched_peaks=unmatched_peaks,
            matched_peak_indices=matched_peak_indices,
            unmatched_ion_labels=tuple(
                sorted(
                    ion.label
                    for ion in ions
                    if ion.label not in matched_ions
                )
            ),
            ion_count=len(ions),
            matched_ion_count=len(matched_ions),
            explanations_truncated=truncated,
            candidate=candidate,
        ),
        truncated,
    )


def _match_exclusive(
    candidates,
    ion_sets,
    peaks,
    intensity_factors,
    tolerance,
    tolerance_unit,
    max_explanations,
):
    """Jointly assign each peak to at most one candidate/ion."""
    flat_ions = []
    for candidate_index, ions in enumerate(ion_sets):
        for ion_index, ion in enumerate(ions):
            flat_ions.append((candidate_index, ion_index, ion))

    edges = []
    for peak in peaks:
        for node_index, (_candidate_index, _ion_index, ion) in enumerate(flat_ions):
            if _within_tolerance(peak.mz, ion.mz, tolerance, tolerance_unit):
                weight = _peak_edge_weight(
                    peak, ion, intensity_factors[peak.index]
                )
                if weight > 0:
                    edges.append((peak.index, node_index, weight))

    peak_ids = tuple(peak.index for peak in peaks)
    node_ids = tuple(range(len(flat_ions)))
    explanations, _best, truncated = _enumerate_matchings(
        peak_ids,
        node_ids,
        edges,
        max_explanations if max_explanations is not None else 10**9,
    )

    global_explanations = []
    candidate_explanations = [[] for _candidate in candidates]
    for matching, score, flag in explanations:
        global_assignments = []
        per_candidate = defaultdict(list)
        for peak_index, node_index in matching:
            candidate_index, _ion_index, ion = flat_ions[node_index]
            peak = peaks[peak_index]
            error = peak.mz - ion.mz
            assignment = Assignment(
                peak_index=peak.index,
                peak_mz=peak.mz,
                peak_intensity=peak.intensity,
                ion_label=ion.label,
                ion_mz=ion.mz,
                error=error,
                error_ppm=(error / ion.mz * 1_000_000.0) if ion.mz else 0.0,
                weight=ion.weight,
                score=_peak_edge_weight(
                    peak, ion, intensity_factors[peak_index]
                ),
            )
            global_assignments.append(
                GlobalAssignment(
                    candidate_index=candidate_index,
                    candidate_name=candidates[candidate_index].name,
                    assignment=assignment,
                )
            )
            per_candidate[candidate_index].append(assignment)
        global_assignments.sort(
            key=lambda item: (
                item.assignment.peak_index,
                item.candidate_index,
                item.assignment.ion_label,
            )
        )
        global_explanations.append(
            GlobalExplanation(
                assignments=tuple(global_assignments),
                score=score,
                truncated=flag,
            )
        )
        for candidate_index in range(len(candidates)):
            assignments = tuple(
                sorted(
                    per_candidate.get(candidate_index, ()),
                    key=lambda item: (item.peak_index, item.ion_label),
                )
            )
            candidate_explanations[candidate_index].append(
                (assignments, sum(item.score for item in assignments))
            )

    candidate_matches = []
    for candidate_index, candidate in enumerate(candidates):
        ions = ion_sets[candidate_index]
        unique = []
        seen = set()
        for explanation, explanation_score in candidate_explanations[candidate_index]:
            key = tuple(
                (item.peak_index, item.ion_label, round(item.score, 12))
                for item in explanation
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append((explanation, explanation_score))
        unique.sort(
            key=lambda item: (
                -round(item[1], 12),
                tuple(
                    (assignment.peak_index, assignment.ion_label)
                    for assignment in item[0]
                ),
            )
        )
        if max_explanations is not None:
            unique = unique[:max_explanations]
        if not unique:
            unique = [((), 0.0)]
        primary, candidate_score = unique[0]
        explanation_tuples = tuple(item[0] for item in unique)
        matched_peak_indices = tuple(
            sorted({assignment.peak_index for assignment in primary})
        )
        matched_peak_set = set(matched_peak_indices)
        unmatched_peaks = tuple(
            peak for peak in peaks if peak.index not in matched_peak_set
        )
        matched_ions = {assignment.ion_label for assignment in primary}
        total_weight = sum(ion.weight for ion in ions)
        matched_weight = sum(assignment.weight for assignment in primary)
        candidate_matches.append(
            CandidateMatch(
                name=candidate.name,
                index=candidate_index,
                score=candidate_score,
                matched_weight=matched_weight,
                total_weight=total_weight,
                coverage=(len(matched_ions) / len(ions)) if ions else 0.0,
                support=(matched_weight / total_weight) if total_weight else 0.0,
                assignments=primary,
                explanations=explanation_tuples,
                unmatched_peaks=unmatched_peaks,
                matched_peak_indices=matched_peak_indices,
                unmatched_ion_labels=tuple(
                    sorted(
                        ion.label for ion in ions if ion.label not in matched_ions
                    )
                ),
                ion_count=len(ions),
                matched_ion_count=len(matched_ions),
                explanations_truncated=truncated,
                candidate=candidate,
            )
        )

    used_peak_indices = set()
    for explanation in global_explanations:
        used_peak_indices.update(
            item.assignment.peak_index for item in explanation.assignments
        )
    unmatched = tuple(
        peak for peak in peaks if peak.index not in used_peak_indices
    )

    return MatchResult(
        candidates=tuple(candidate_matches),
        peaks=peaks,
        unmatched_peaks=unmatched,
        peak_usage="exclusive",
        tolerance=tolerance,
        tolerance_unit=tolerance_unit,
        global_explanations=tuple(global_explanations),
        explanations_truncated=truncated,
    )


class FragmentMatcher:
    """
    Reusable fragment matcher with fixed matching options.

    Arguments are the same as for :func:`match_fragments`, except that the
    spectrum and candidates are supplied to :meth:`match`.
    """

    def __init__(
        self,
        neutral_losses=None,
        *,
        tolerance=None,
        tolerance_unit=None,
        weights=None,
        intensity_weighting="none",
        peak_usage="per_candidate",
        max_explanations=100,
        spectrum_peak_type="centroided",
        min_intensity=None,
        **options,
    ):
        self.neutral_losses = _normalise_losses(neutral_losses)
        self.tolerance = tolerance
        self.tolerance_unit = tolerance_unit
        self.weights = weights
        self.intensity_weighting = intensity_weighting
        self.peak_usage = peak_usage
        self.max_explanations = max_explanations
        self.spectrum_peak_type = spectrum_peak_type
        self.min_intensity = min_intensity
        self.options = _resolve_options(options)

    def match(self, spectrum, candidates, neutral_losses=None, **options):
        """Run the matcher on one spectrum and one or more candidates."""
        losses = neutral_losses
        if losses is None:
            losses = self.neutral_losses
        merged = dict(self.options)
        merged.update(options)
        return match_fragments(
            spectrum,
            candidates,
            losses,
            tolerance=merged.pop("tolerance", self.tolerance),
            tolerance_unit=merged.pop("tolerance_unit", self.tolerance_unit),
            weights=merged.pop("weights", self.weights),
            intensity_weighting=merged.pop(
                "intensity_weighting", self.intensity_weighting
            ),
            peak_usage=merged.pop("peak_usage", self.peak_usage),
            max_explanations=merged.pop(
                "max_explanations", self.max_explanations
            ),
            spectrum_peak_type=merged.pop(
                "spectrum_peak_type", self.spectrum_peak_type
            ),
            min_intensity=merged.pop("min_intensity", self.min_intensity),
            **merged,
        )

    __call__ = match

    def match_candidates(self, candidates, spectrum, neutral_losses=None, **options):
        """Alias of :meth:`match` with the candidates first."""
        return self.match(spectrum, candidates, neutral_losses, **options)


# Common alternative spellings for the same operation.
def match_candidates(candidates, spectrum, *args, **kwargs):
    """Candidates-first alias of :func:`match_fragments`."""
    return match_fragments(spectrum, candidates, *args, **kwargs)


match_spectrum = match_fragments
