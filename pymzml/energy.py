#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collision energy handling for mzML spectra.

Scan energy fields are not interchangeable:

* ``MS:1000045`` (collision energy) carries one energy value in electronvolt
  by convention,
* ``MS:1000138`` (percent collision energy) carries a relative value and is
  unitless; it cannot be converted to electronvolt without instrument-specific
  calibration,
* ``MS:1002013``/``MS:1002014`` (collision energy ramp start/end) describe a
  continuous mixture of energies between two values,
* repeated ``MS:1000045`` values (or a list of values in a vendor
  ``userParam``) describe a stepped mixture,
* ``MS:1000509`` (activation energy) is a separate precursor activation
  attribute and is not a collision energy,
* vendor ``userParam`` and filter-string values have no CV semantics and must
  therefore carry their own provenance.

:class:`AcquisitionEnergy` keeps the covered range, the discrete steps when
they are known, and every source that contributed the value.  Equality is
exact: a single energy inside a mixed range is *not* the same acquisition
condition as the mixture.  :func:`bucket_conditions` distinguishes energies a
spectrum was certainly acquired at from energies it only possibly covered.
"""

import math
import re

# MS controlled vocabulary accessions.
COLLISION_ENERGY = "MS:1000045"
PERCENT_COLLISION_ENERGY = "MS:1000138"
ACTIVATION_ENERGY = "MS:1000509"
COLLISION_ENERGY_RAMP_START = "MS:1002013"
COLLISION_ENERGY_RAMP_END = "MS:1002014"

EV = "electronvolt"
JOULE = "joule"
PERCENT = "percent"

# Unit Ontology accessions used by collision energy parameters.
UNIT_ACCESSIONS = {
    "UO:0000266": EV,
    "UO:0000111": JOULE,
    "UO:0000112": JOULE,
    "UO:0000187": PERCENT,
    "MS:1000137": EV,
}

UNIT_NAMES = {
    "electronvolt": EV,
    "electron volt": EV,
    "electronvolts": EV,
    "electron volts": EV,
    "ev": EV,
    "joule": JOULE,
    "joules": JOULE,
    "j": JOULE,
    "percent": PERCENT,
    "percentage": PERCENT,
    "%": PERCENT,
}

# Sources, strongest first.
CV_SOURCE = "cvParam"
USER_PARAM_SOURCE = "userParam"
FILTER_STRING_SOURCE = "filter string"

_JOULE_PER_EV = 1.602176634e-19

_MULTI_VALUE_SEPARATORS = re.compile(r"\s*[,;/|]\s*|\s+")
_RANGE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_FILTER_ENERGY_TOKEN = re.compile(
    r"@(?:cid|hcd|ce)(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class EnergySource(object):
    """
    One place a collision energy value was recorded.

    Arguments:
        kind (str): ``"cvParam"``, ``"userParam"`` or ``"filter string"``
        accession (str): CV accession, when available
        name (str): original parameter name
        raw_value (str): original textual value
        unit (str): unit as declared by this source (may be ``None``)
    """

    __slots__ = ("kind", "accession", "name", "raw_value", "unit")

    def __init__(self, kind, accession=None, name=None, raw_value=None, unit=None):
        self.kind = kind
        self.accession = accession
        self.name = name
        self.raw_value = raw_value
        self.unit = unit

    def __eq__(self, other):
        if not isinstance(other, EnergySource):
            return NotImplemented
        return (
            self.kind,
            self.accession,
            self.name,
            self.raw_value,
            self.unit,
        ) == (
            other.kind,
            other.accession,
            other.name,
            other.raw_value,
            other.unit,
        )

    def __hash__(self):
        return hash((self.kind, self.accession, self.name, self.raw_value, self.unit))

    def __repr__(self):
        return "EnergySource({0!r}, accession={1!r}, value={2!r})".format(
            self.kind, self.accession, self.raw_value
        )

    def as_dict(self):
        return {
            "kind": self.kind,
            "accession": self.accession,
            "name": self.name,
            "raw_value": self.raw_value,
            "unit": self.unit,
        }


class AcquisitionEnergy(object):
    """
    A comparable acquisition energy condition.

    Arguments:
        low (float): lowest energy in the condition
        high (float): highest energy in the condition; defaults to ``low``
        unit (str or None): ``"electronvolt"``, ``"joule"``, ``"percent"`` or
            ``None`` when the source declares no unit
        sources (iterable): :class:`EnergySource` entries
        steps (iterable): discrete energies when the mixture is stepped

    The condition is a single energy when ``low == high`` and a mixture
    otherwise.  ``steps`` is ``None`` for a continuous ramp and a sorted tuple
    for a stepped mixture.

    A unit of ``None`` means *unknown*, not electronvolt.  Unknown-unit
    conditions can be compared with each other and bucketed among themselves,
    but converting them requires an explicit assumption (see
    :meth:`comparable_conditions`'s ``assume_unit``).
    """

    __slots__ = ("low", "high", "unit", "sources", "steps", "continuous")

    def __init__(
        self,
        low=None,
        high=None,
        unit=EV,
        sources=None,
        steps=None,
        continuous=False,
        **kwargs
    ):
        if low is None and "value" in kwargs:
            low = kwargs.pop("value")
        if kwargs:
            raise TypeError(
                "Unexpected keyword arguments: {0}".format(
                    ", ".join(sorted(kwargs))
                )
            )
        if low is None:
            raise TypeError("An energy value is required")
        low = float(low)
        if high is None:
            high = low
        high = float(high)
        if high < low:
            low, high = high, low
        self.low = low
        self.high = high
        self.unit = unit
        self.sources = tuple(sources or ())
        self.continuous = bool(continuous)
        if steps is None:
            self.steps = None
        else:
            unique = sorted({float(step) for step in steps})
            self.steps = tuple(unique)

    # -- shape -----------------------------------------------------------

    @property
    def is_exact(self):
        """Whether the condition is one fixed energy."""
        return self.low == self.high and not self.continuous

    @property
    def is_fixed(self):
        """Alias for :attr:`is_exact`."""
        return self.is_exact

    @property
    def is_range(self):
        """Whether the condition covers a mixture of energies."""
        return not self.is_exact

    @property
    def is_mixture(self):
        """Alias for :attr:`is_range`."""
        return self.is_range

    @property
    def is_stepped(self):
        """Whether the mixture is known to use discrete energy steps."""
        return self.steps is not None and len(self.steps) > 1

    @property
    def is_stepwise(self):
        """Alias for :attr:`is_stepped`."""
        return self.is_stepped

    @property
    def is_discrete(self):
        """Whether the condition covers only discrete, explicit energies."""
        return self.steps is not None

    @property
    def has_steps(self):
        """Alias for :attr:`is_stepped`."""
        return self.is_stepped

    @property
    def value(self):
        """
        The single energy value.

        Raises:
            ValueError: if the condition is a mixture.
        """
        if self.is_range:
            raise ValueError(
                "Collision energy is a range ({0}-{1}); use low/high".format(
                    self.low, self.high
                )
            )
        return self.low

    @property
    def range(self):
        """The ``(low, high)`` interval covered by this condition."""
        return (self.low, self.high)

    @property
    def min(self):
        """Lowest energy in this condition."""
        return self.low

    @property
    def max(self):
        """Highest energy in this condition."""
        return self.high

    @property
    def minimum(self):
        """Alias for :attr:`low`."""
        return self.low

    @property
    def maximum(self):
        """Alias for :attr:`high`."""
        return self.high

    @property
    def step_values(self):
        """Alias for :attr:`steps`."""
        return self.steps

    @property
    def discrete_values(self):
        """Alias for :attr:`steps`."""
        return self.steps

    @property
    def lower(self):
        """Alias for :attr:`low`."""
        return self.low

    @property
    def upper(self):
        """Alias for :attr:`high`."""
        return self.high

    @property
    def unit_name(self):
        """Alias for :attr:`unit`."""
        return self.unit

    @property
    def has_unknown_unit(self):
        """Whether the unit is unknown for this condition."""
        return self.unit is None

    @property
    def unit_known(self):
        """Whether the source declared a unit for this condition."""
        return self.unit is not None

    @property
    def unit_is_known(self):
        """Alias for :attr:`unit_known`."""
        return self.unit_known

    @property
    def kind(self):
        """
        Shape of this condition.

        Returns:
            str: ``"fixed"``, ``"stepped"``, ``"ramp"`` or ``"range"``
        """
        if self.is_exact:
            return "fixed"
        if self.is_stepped:
            return "stepped"
        if self.continuous:
            return "ramp"
        return "range"

    @property
    def values(self):
        """The covered interval as ``(low, high)``."""
        return (self.low, self.high)

    @property
    def energy_values(self):
        """Sorted discrete energies when stepped, else the covered interval."""
        if self.steps is not None:
            return self.steps
        return self.values

    @property
    def source(self):
        """The first source kind, for convenient display."""
        return self.sources[0].kind if self.sources else None

    @property
    def primary_source(self):
        """The strongest recorded source, preferring CV terms."""
        for source in self.sources:
            if source.kind == CV_SOURCE:
                return source
        return self.sources[0] if self.sources else None

    @property
    def source_kinds(self):
        """All distinct source kinds, in the order they were recorded."""
        seen = []
        for source in self.sources:
            if source.kind not in seen:
                seen.append(source.kind)
        return tuple(seen)

    @property
    def source_types(self):
        """Alias for :attr:`source_kinds`."""
        return self.source_kinds

    @property
    def accessions(self):
        """All distinct CV accessions of the contributing sources."""
        seen = []
        for source in self.sources:
            if source.accession is not None and source.accession not in seen:
                seen.append(source.accession)
        return tuple(seen)

    @property
    def source_names(self):
        """Parameter names of all contributing sources."""
        return tuple(
            source.name for source in self.sources if source.name is not None
        )

    @property
    def raw_values(self):
        """Original textual values of all contributing sources."""
        return tuple(
            source.raw_value
            for source in self.sources
            if source.raw_value is not None
        )

    _MAPPING_KEYS = (
        "low",
        "high",
        "value",
        "range",
        "unit",
        "unit_name",
        "unit_known",
        "unit_is_known",
        "has_unknown_unit",
        "is_exact",
        "is_fixed",
        "is_range",
        "is_mixture",
        "is_stepped",
        "is_stepwise",
        "is_discrete",
        "steps",
        "step_values",
        "discrete_values",
        "kind",
        "source",
        "sources",
        "source_kinds",
        "source_types",
        "source_names",
        "accessions",
        "raw_values",
    )

    def __getitem__(self, key):
        if key not in self._MAPPING_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        """Mapping-style access with a default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """List the keys accepted by mapping-style access."""
        return self._MAPPING_KEYS

    def to_dict(self):
        """Alias for :meth:`as_dict`."""
        return self.as_dict()

    def to(self, unit, assume_unit=None):
        """Return this condition converted to ``unit``.

        Conditions whose source declared no unit cannot be converted unless
        ``assume_unit`` states which unit the source convention uses.
        """
        source_unit = self.unit if self.unit is not None else assume_unit
        if source_unit is None:
            raise ValueError(
                "Energy unit is unknown; pass assume_unit to convert it"
            )
        if source_unit == unit:
            return AcquisitionEnergy(
                self.low,
                self.high,
                unit=unit,
                sources=self.sources,
                steps=self.steps,
                continuous=self.continuous,
            )
        if source_unit == PERCENT or unit == PERCENT:
            raise ValueError(
                "Percent collision energy cannot be converted to {0!r}".format(unit)
            )
        return AcquisitionEnergy(
            convert_unit(self.low, source_unit, unit),
            convert_unit(self.high, source_unit, unit),
            unit=unit,
            sources=self.sources,
            steps=(
                tuple(convert_unit(step, source_unit, unit) for step in self.steps)
                if self.steps is not None
                else None
            ),
            continuous=self.continuous,
        )

    def converted(self, unit, assume_unit=None):
        """Alias for :meth:`to`."""
        return self.to(unit, assume_unit=assume_unit)

    # -- comparisons -----------------------------------------------------

    def contains(self, energy, unit=None):
        """
        Whether a single energy value lies within this condition.

        This is interval membership only.  It does not say that the two
        conditions are equal: a mixture may contain many single energies.
        """
        if isinstance(energy, AcquisitionEnergy):
            return self.covers(energy)
        energy = float(energy)
        if unit is not None and unit != self.unit:
            if self.unit is None:
                raise ValueError(
                    "Energy unit is unknown; cannot test unit-specific membership"
                )
            energy = convert_unit(energy, unit, self.unit)
        if self.steps is not None:
            return any(step == energy for step in self.steps)
        return self.low <= energy <= self.high

    def includes(self, energy, unit=None):
        """Alias for :meth:`contains`."""
        return self.contains(energy, unit=unit)

    def covers(self, other):
        """Whether every energy of ``other`` lies in this condition."""
        if not isinstance(other, AcquisitionEnergy):
            return NotImplemented
        if self.unit != other.unit:
            raise TypeError("Cannot compare energies with different units")
        return self.low <= other.low and other.high <= self.high

    def matches(self, other):
        """Whether two conditions cover exactly the same energies."""
        if not isinstance(other, AcquisitionEnergy):
            if isinstance(other, (int, float)):
                return self.is_exact and self.low == other
            if isinstance(other, tuple) and len(other) == 2:
                return (
                    self.is_range
                    and self.low == float(other[0])
                    and self.high == float(other[1])
                )
            return NotImplemented
        if self.unit != other.unit:
            return False
        return (
            self.low == other.low
            and self.high == other.high
            and self.steps == other.steps
            and self.continuous == other.continuous
        )

    def same_condition(self, other):
        """Alias for :meth:`matches`."""
        return self.matches(other)

    def same_as(self, other):
        """Alias for :meth:`matches`."""
        return self.matches(other)

    def same(self, other):
        """Alias for :meth:`matches`."""
        return self.matches(other)

    def overlaps(self, other):
        """Whether two conditions share at least one energy."""
        if not isinstance(other, AcquisitionEnergy):
            return NotImplemented
        if self.unit != other.unit:
            raise TypeError("Cannot compare energies with different units")
        return self.low <= other.high and other.low <= self.high

    def _key(self):
        return (self.unit, self.low, self.high, self.steps, self.continuous)

    def __eq__(self, other):
        if isinstance(other, AcquisitionEnergy):
            return self._key() == other._key()
        if isinstance(other, (int, float)):
            return self.is_exact and self.low == other
        if isinstance(other, tuple) and len(other) == 2:
            return (
                self.is_range
                and self.low == float(other[0])
                and self.high == float(other[1])
            )
        return NotImplemented

    def __hash__(self):
        return hash(self._key())

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)

    def __lt__(self, other):
        if isinstance(other, (int, float)) and self.is_exact:
            return self.low < other
        if isinstance(other, AcquisitionEnergy):
            if self.unit != other.unit:
                raise TypeError("Cannot order energies with different units")
            return (self.low, self.high) < (other.low, other.high)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (int, float)) and self.is_exact:
            return self.low <= other
        if isinstance(other, AcquisitionEnergy):
            return self.low == other.low and self.high == other.high
        return NotImplemented

    def __repr__(self):
        return "AcquisitionEnergy({0}, unit={1!r}, sources={2!r})".format(
            self.to_string(), self.unit, [source.kind for source in self.sources]
        )

    def to_string(self):
        """Human readable representation, e.g. ``"25-40 electronvolt"``."""
        unit = self.unit if self.unit is not None else "unknown unit"
        if self.is_exact:
            body = "{0:g} {1}".format(self.low, unit)
        else:
            body = "{0:g}-{1:g} {2}".format(self.low, self.high, unit)
        if self.is_stepped:
            body += " (steps: {0})".format(
                ", ".join("{0:g}".format(step) for step in self.steps)
            )
        return body

    def as_dict(self):
        """Return a plain dictionary with all fields of this condition."""
        return {
            "low": self.low,
            "high": self.high,
            "range": self.range,
            "unit": self.unit,
            "unit_known": self.unit_known,
            "is_exact": self.is_exact,
            "is_range": self.is_range,
            "is_stepped": self.is_stepped,
            "is_discrete": self.is_discrete,
            "continuous": self.continuous,
            "kind": self.kind,
            "steps": self.steps,
            "sources": [source.as_dict() for source in self.sources],
            "source_kinds": list(self.source_kinds),
            "accessions": list(self.accessions),
        }


# Backwards-friendly descriptive alias.
CollisionEnergy = AcquisitionEnergy
EnergyCondition = AcquisitionEnergy


class EnergyBucket(object):
    """
    One energy bucket touched by an acquisition condition.

    Arguments:
        start (float): inclusive lower bound of the bucket
        end (float): exclusive upper bound of the bucket
        unit (str): energy unit
        definite (bool): a fixed energy of the acquisition falls in this bucket
        possible (bool): a mixture condition covers part of this bucket

    ``definite`` and ``possible`` can both be true for one bucket when a
    spectrum has a fixed energy and a stepping/ramp mixture covering it.
    """

    __slots__ = ("start", "end", "unit", "definite", "possible", "sources")

    def __init__(
        self,
        start,
        end,
        unit=EV,
        definite=False,
        possible=False,
        sources=None,
    ):
        self.start = float(start)
        self.end = float(end)
        self.unit = unit
        self.definite = bool(definite)
        self.possible = bool(possible)
        self.sources = tuple(sources or ())

    @property
    def center(self):
        return (self.start + self.end) / 2.0

    @property
    def energy(self):
        """Bucket lower bound, which is also the bucket's identity."""
        return self.start

    @property
    def low(self):
        """Alias for :attr:`start`."""
        return self.start

    @property
    def high(self):
        """Alias for :attr:`end`."""
        return self.end

    @property
    def membership(self):
        """``"definite"`` if certain, else ``"possible"`` if only covered."""
        if self.definite:
            return "definite"
        return "possible" if self.possible else "none"

    @property
    def certain(self):
        """Alias for :attr:`definite`."""
        return self.definite

    @property
    def covered(self):
        """Alias for :attr:`possible`."""
        return self.possible

    @property
    def source_kinds(self):
        """Distinct source kinds recorded for this bucket."""
        seen = []
        for source in self.sources:
            if source.kind not in seen:
                seen.append(source.kind)
        return tuple(seen)

    _MAPPING_KEYS = (
        "start",
        "end",
        "low",
        "high",
        "center",
        "energy",
        "unit",
        "definite",
        "possible",
        "is_definite",
        "is_possible",
        "certain",
        "covered",
        "membership",
        "sources",
        "source_kinds",
    )

    def __getitem__(self, key):
        if key not in self._MAPPING_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        """Mapping-style access with a default."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """List the keys accepted by mapping-style access."""
        return self._MAPPING_KEYS

    def to_dict(self):
        """Alias for :meth:`as_dict`."""
        return self.as_dict()

    @property
    def is_definite(self):
        return self.definite

    @property
    def is_possible(self):
        return self.possible

    def __eq__(self, other):
        if not isinstance(other, EnergyBucket):
            return NotImplemented
        return (
            self.start,
            self.end,
            self.unit,
            self.definite,
            self.possible,
        ) == (
            other.start,
            other.end,
            other.unit,
            other.definite,
            other.possible,
        )

    def __hash__(self):
        return hash((self.start, self.end, self.unit, self.definite, self.possible))

    def __repr__(self):
        return "EnergyBucket({0:g}-{1:g} {2}, definite={3}, possible={4})".format(
            self.start, self.end, self.unit, self.definite, self.possible
        )

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
            "definite": self.definite,
            "possible": self.possible,
            "membership": self.membership,
            "sources": [source.as_dict() for source in self.sources],
            "source_kinds": list(self.source_kinds),
        }


# -- parsing helpers ------------------------------------------------------


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_unit(unit_name, unit_accession=None):
    if unit_accession and unit_accession in UNIT_ACCESSIONS:
        return UNIT_ACCESSIONS[unit_accession]
    if unit_name:
        key = " ".join(unit_name.strip().lower().split())
        if key in UNIT_NAMES:
            return UNIT_NAMES[key]
    return None


def convert_unit(value, from_unit, to_unit):
    """
    Convert an energy value between electronvolt and joule.

    Percent collision energy is relative and has no fixed conversion to
    electronvolt; attempting it raises ``ValueError`` instead of silently
    producing incomparable numbers.
    """
    if from_unit == to_unit:
        return value
    if from_unit == EV and to_unit == JOULE:
        return value * _JOULE_PER_EV
    if from_unit == JOULE and to_unit == EV:
        return value / _JOULE_PER_EV
    raise ValueError(
        "Cannot convert energy from {0!r} to {1!r}".format(from_unit, to_unit)
    )


def _split_values(raw_value):
    """
    Split a textual value into one or more numbers.

    Handles comma/semicolon lists produced by stepped-energy instruments while
    leaving decimal points intact.
    """
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if text == "":
        return []
    pieces = [piece for piece in _MULTI_VALUE_SEPARATORS.split(text) if piece]
    values = []
    for piece in pieces:
        value = _parse_float(piece)
        if value is not None:
            values.append(value)
    return values


def _condition_from_pieces(pieces, unit, source, steps=None):
    if not pieces:
        return None
    low, high = min(pieces), max(pieces)
    if steps is None and len(pieces) > 1:
        steps = pieces
    return AcquisitionEnergy(low, high, unit=unit, sources=[source], steps=steps)


def _cv_source(element):
    return EnergySource(
        CV_SOURCE,
        accession=element.get("accession"),
        name=element.get("name"),
        raw_value=element.get("value"),
        unit=_normalize_unit(element.get("unitName"), element.get("unitAccession")),
    )


def _cv_condition(element, default_unit=EV):
    accession = element.get("accession")
    text = (element.get("value") or "").strip()
    values = _split_values(text)

    if accession == PERCENT_COLLISION_ENERGY:
        unit = PERCENT
    else:
        declared_unit = _normalize_unit(
            element.get("unitName"), element.get("unitAccession")
        )
        unit = declared_unit if declared_unit is not None else default_unit

    source = _cv_source(element)
    source.unit = unit

    range_match = _RANGE_PATTERN.match(text)
    if range_match:
        return AcquisitionEnergy(
            float(range_match.group(1)),
            float(range_match.group(2)),
            unit=unit,
            sources=[source],
        )
    if not values:
        return None
    return _condition_from_pieces(values, unit, source)


def parse_activation_element(activation):
    """
    Read collision energy conditions from one mzML ``<activation>`` element.

    Ramp start/end are combined into one continuous condition.  Multiple
    fixed collision energies of the same unit are combined into one stepped
    condition, preserving every value as a step.  ``activation energy``
    (``MS:1000509``) is intentionally ignored because it is not a collision
    energy.
    """
    exact_by_unit = {}
    ramp_start = None
    ramp_end = None

    for element in activation.iter():
        if element.tag.rsplit("}", 1)[-1] != "cvParam":
            continue
        accession = element.get("accession")
        if accession == COLLISION_ENERGY_RAMP_START:
            ramp_start = _cv_condition(element)
        elif accession == COLLISION_ENERGY_RAMP_END:
            ramp_end = _cv_condition(element)
        elif accession in (COLLISION_ENERGY, PERCENT_COLLISION_ENERGY):
            condition = _cv_condition(element)
            if condition is None:
                continue
            exact_by_unit.setdefault(condition.unit, []).append(condition)

    conditions = []

    for unit, group in exact_by_unit.items():
        steps = []
        sources = []
        single_ranges = []
        for condition in group:
            if condition.continuous:
                single_ranges.append(condition)
                continue
            if condition.steps is not None:
                steps.extend(condition.steps)
            else:
                steps.append(condition.low)
            sources.extend(condition.sources)
        if steps:
            conditions.append(
                AcquisitionEnergy(
                    min(steps),
                    max(steps),
                    unit=unit,
                    sources=sources,
                    steps=steps if len(set(steps)) > 1 else None,
                )
            )
        conditions.extend(single_ranges)

    if ramp_start is not None and ramp_end is not None:
        unit = ramp_start.unit
        low = convert_unit(ramp_start.low, ramp_start.unit, unit)
        high = convert_unit(ramp_end.high, ramp_end.unit, unit)
        conditions.append(
            AcquisitionEnergy(
                low,
                high,
                unit=unit,
                sources=ramp_start.sources + ramp_end.sources,
                continuous=True,
            )
        )
    elif ramp_start is not None:
        conditions.append(_as_ramp_endpoint(ramp_start))
    elif ramp_end is not None:
        conditions.append(_as_ramp_endpoint(ramp_end))

    return conditions


def _as_ramp_endpoint(condition):
    """
    Rebuild a lone ramp start/end as a one-sided continuous condition.

    A ramp with only one recorded end point is still a ramp; its full extent
    is unknown, so it must not be reported as a fixed energy.
    """
    return AcquisitionEnergy(
        condition.low,
        condition.high,
        unit=condition.unit,
        sources=condition.sources,
        continuous=True,
    )


def _user_param_condition(element):
    unit = _normalize_unit(element.get("unitName"), element.get("unitAccession"))
    source = EnergySource(
        USER_PARAM_SOURCE,
        name=element.get("name"),
        raw_value=element.get("value"),
        unit=unit,
    )

    text = (element.get("value") or "").strip()
    range_match = _RANGE_PATTERN.match(text)
    if range_match:
        return AcquisitionEnergy(
            float(range_match.group(1)),
            float(range_match.group(2)),
            unit=unit,
            sources=[source],
        )

    values = _split_values(text)
    if not values:
        return None
    return _condition_from_pieces(values, unit, source)


def _ramp_user_param(element):
    unit = _normalize_unit(element.get("unitName"), element.get("unitAccession"))
    values = _split_values(element.get("value"))
    if not values:
        return None
    source = EnergySource(
        USER_PARAM_SOURCE,
        name=element.get("name"),
        raw_value=element.get("value"),
        unit=unit,
    )
    return AcquisitionEnergy(values[0], unit=unit, sources=[source])


def _parse_filter_string(value):
    values = [float(match) for match in _FILTER_ENERGY_TOKEN.findall(value or "")]
    if not values:
        return []
    # ``cid35`` in a vendor filter string is normalized/relative collision
    # energy, not electronvolt, so no unit is claimed.
    source = EnergySource(
        FILTER_STRING_SOURCE,
        name="filter string",
        raw_value=value,
        unit=None,
    )
    if len(values) > 1:
        return [
            AcquisitionEnergy(
                min(values),
                max(values),
                unit=None,
                sources=[source],
                steps=values,
            )
        ]
    return [AcquisitionEnergy(values[0], unit=None, sources=[source])]


def filter_string_conditions(spectrum_element):
    """
    Extract collision energies from vendor ``filter string`` userParams.

    Filter strings carry no CV semantics, so this is opt-in and every result
    keeps its ``filter string`` provenance.
    """
    conditions = []
    for element in spectrum_element.iter():
        if element.tag.rsplit("}", 1)[-1] != "userParam":
            continue
        if (element.get("name") or "").lower() != "filter string":
            continue
        conditions.extend(_parse_filter_string(element.get("value")))
    return conditions


def _merge_conditions(conditions):
    """
    Merge conditions that cover exactly the same interval and unit.

    Sources are always preserved.  A vendor condition whose unit is unknown is
    merged into an otherwise identical unit-known condition: the CV/declared
    unit wins and the vendor source is retained as corroboration.  A stepped
    mixture is only merged with the same steps.  Non-identical intervals are
    never merged; overlapping ranges remain separate conditions.
    """
    merged = []
    for condition in conditions:
        merged = _merge_into(merged, condition)
    return merged


def _merge_into(merged, condition):
    for index, existing in enumerate(merged):
        if not _same_interval(existing, condition):
            continue
        known_unit = (
            existing.unit if existing.unit is not None else condition.unit
        )
        combined_sources = existing.sources + tuple(
            source
            for source in condition.sources
            if source not in existing.sources
        )
        merged[index] = AcquisitionEnergy(
            existing.low,
            existing.high,
            unit=known_unit,
            sources=combined_sources,
            steps=existing.steps,
            continuous=existing.continuous,
        )
        return merged
    return merged + [condition]


def _same_interval(first, second):
    if first.low != second.low or first.high != second.high:
        return False
    if first.steps != second.steps:
        return False
    if first.continuous != second.continuous:
        return False
    if first.unit is None or second.unit is None:
        return True
    return first.unit == second.unit


def get_energy_conditions(element, include_filter_string=True):
    """
    Module-level convenience wrapper around :func:`parse_spectrum_element`.

    Accepts either a ``<spectrum>`` element or an object exposing ``element``
    (such as :class:`pymzml.spec.Spectrum`).
    """
    if hasattr(element, "element"):
        element = element.element
    if element is None:
        return []
    return parse_spectrum_element(
        element, include_filter_string=include_filter_string
    )


def parse_spectrum_element(spectrum_element, include_filter_string=True):
    """
    Collect the collision energy conditions of a spectrum.

    Arguments:
        spectrum_element (xml.etree.ElementTree.Element): ``<spectrum>`` node

    Keyword Arguments:
        include_filter_string (bool): also read vendor ``filter string``
            values.  These are heuristics without CV semantics, so they are
            marked with the ``filter string`` source.

    Returns:
        list: :class:`AcquisitionEnergy` conditions.  Conditions that are
        numerically identical are merged and keep all contributing sources.
    """
    conditions = []
    ramp_start = None
    ramp_end = None
    for element in spectrum_element.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "activation":
            conditions.extend(parse_activation_element(element))
        elif tag == "userParam":
            name = (element.get("name") or "").strip().lower()
            if name == "collision energy":
                condition = _user_param_condition(element)
                if condition is not None:
                    conditions.append(condition)
            elif name == "collision energy ramp start":
                ramp_start = _ramp_user_param(element)
            elif name == "collision energy ramp end":
                ramp_end = _ramp_user_param(element)

    if ramp_start is not None and ramp_end is not None:
        conditions.append(
            AcquisitionEnergy(
                min(ramp_start.low, ramp_end.low),
                max(ramp_start.high, ramp_end.high),
                unit=ramp_start.unit,
                sources=ramp_start.sources + ramp_end.sources,
                continuous=True,
            )
        )
    elif ramp_start is not None:
        conditions.append(_as_ramp_endpoint(ramp_start))
    elif ramp_end is not None:
        conditions.append(_as_ramp_endpoint(ramp_end))
    if include_filter_string:
        conditions.extend(filter_string_conditions(spectrum_element))
    return _merge_conditions(conditions)


def merge_ranges(conditions):
    """
    Merge overlapping conditions of the same unit into wider ranges.

    Provenance is kept by merging sources.  Conditions from different units
    are never merged.  ``steps`` are dropped when merging different stepped
    ranges because the combined mixture is no longer one discrete step set.
    """
    groups = {}
    for condition in conditions:
        groups.setdefault(condition.unit, []).append(condition)
    merged = []
    for unit, group in groups.items():
        ordered = sorted(group, key=lambda condition: (condition.low, condition.high))
        for condition in ordered:
            if merged and merged[-1].unit == unit and condition.low <= merged[-1].high:
                current = merged[-1]
                steps = None
                if (
                    current.steps is not None
                    and condition.steps is not None
                    and current.low == condition.low
                    and current.high == condition.high
                ):
                    steps = tuple(set(current.steps) | set(condition.steps))
                merged[-1] = AcquisitionEnergy(
                    current.low,
                    max(current.high, condition.high),
                    unit=unit,
                    sources=current.sources + condition.sources,
                    steps=steps,
                    continuous=current.continuous or condition.continuous,
                )
            else:
                merged.append(condition)
    return merged


# -- comparable view ------------------------------------------------------


def comparable_conditions(conditions, unit=None, assume_unit=None):
    """
    Return conditions in one unit, suitable for direct comparison.

    Electronvolt and joule are convertible.  Percent collision energy is left
    unchanged unless it is explicitly requested, because it is a relative
    value with no instrument-independent conversion.

    Arguments:
        conditions (iterable): :class:`AcquisitionEnergy` values
        unit (str or None): target unit; ``None`` keeps native units but still
            merges identical conditions.
        assume_unit (str or None): unit to assign to conditions whose source
            declared none.  Without it, unknown-unit conditions remain
            unknown and cannot be silently combined with eV values.

    Returns:
        list: conditions in ``unit`` (or native units).
    """
    converted = []
    for condition in conditions:
        source_unit = condition.unit
        if source_unit is None:
            source_unit = assume_unit
        if unit is None or source_unit == unit:
            candidate = AcquisitionEnergy(
                condition.low,
                condition.high,
                unit=source_unit,
                sources=condition.sources,
                steps=condition.steps,
                continuous=condition.continuous,
            )
        else:
            if source_unit is None:
                raise ValueError(
                    "Energy unit is unknown; pass assume_unit to compare it"
                )
            if source_unit == PERCENT or unit == PERCENT:
                raise ValueError(
                    "Percent collision energy cannot be converted to {0!r}".format(
                        unit
                    )
                )
            candidate = AcquisitionEnergy(
                convert_unit(condition.low, source_unit, unit),
                convert_unit(condition.high, source_unit, unit),
                unit=unit,
                sources=condition.sources,
                steps=(
                    tuple(
                        convert_unit(step, source_unit, unit)
                        for step in condition.steps
                    )
                    if condition.steps is not None
                    else None
                ),
                continuous=condition.continuous,
            )
        converted.append(candidate)
    return _merge_conditions(converted)


def same_condition(first, second):
    """
    Whether two conditions are exactly the same acquisition condition.

    A single energy inside a mixture returns ``False``; use
    :meth:`AcquisitionEnergy.contains` for interval membership.
    """
    return first.matches(second)


def conditions_equal(first, second):
    """Alias for :func:`same_condition`."""
    return same_condition(first, second)


# -- bucketing ------------------------------------------------------------


def bucket_conditions(
    conditions, width=1.0, unit=None, origin=0.0, assume_unit=None
):
    """
    Assign acquisition conditions to energy buckets.

    A fixed energy is a *definite* member of the bucket containing it.  A
    mixture, whether a continuous ramp or a stepped field, only makes buckets
    *possible*: the spectrum was acquired across several energies, so no
    single bucket can be attributed to it with certainty.  If both a fixed and
    a mixture condition touch a bucket, the bucket is both definite and
    possible.

    Buckets are half-open intervals ``[start, start + width)`` aligned to
    ``origin``.  Stepped mixtures only touch the buckets of their explicit
    steps; continuous ramps touch every bucket they overlap.

    Arguments:
        conditions (iterable): :class:`AcquisitionEnergy` values
        width (float): bucket width
        unit (str or None): target unit; ``None`` keeps native units separate
        origin (float): bucket grid alignment
        assume_unit (str or None): unit to assign to vendor values that
            declared none

    Returns:
        list: :class:`EnergyBucket` entries sorted by unit and start.
    """
    if width <= 0:
        raise ValueError("Bucket width must be positive")

    buckets = {}

    def touch(condition, index, membership):
        key = (condition.unit, index)
        entry = buckets.get(key)
        if entry is None:
            start = origin + index * width
            entry = EnergyBucket(
                start,
                start + width,
                unit=condition.unit,
                definite=False,
                possible=False,
                sources=(),
            )
            buckets[key] = entry
        if membership == "definite":
            entry.definite = True
        else:
            entry.possible = True
        merged_sources = list(entry.sources)
        for source in condition.sources:
            if source not in merged_sources:
                merged_sources.append(source)
        entry.sources = tuple(merged_sources)

    targets = comparable_conditions(
        conditions, unit=unit, assume_unit=assume_unit
    )
    for condition in targets:
        if condition.is_exact:
            touch(condition, _bucket_index(condition.low, width, origin), "definite")
            continue
        if condition.steps is not None:
            for step in condition.steps:
                touch(condition, _bucket_index(step, width, origin), "possible")
            continue
        # Bucket endpoints inclusively: a 20-40 eV range covers the 40 eV
        # boundary energy as well as every bucket in between.
        first = _bucket_index(condition.low, width, origin)
        last = _last_bucket_index(condition.high, width, origin)
        for index in range(first, last + 1):
            touch(condition, index, "possible")

    return sorted(
        buckets.values(),
        key=lambda bucket: (
            bucket.unit is not None,
            bucket.unit or "",
            bucket.start,
        ),
    )


def conditional_buckets(conditions, width=1.0, unit=None, origin=0.0, assume_unit=None):
    """
    Bucket conditions and split them by membership certainty.

    Returns:
        tuple: ``(definite, possible)`` lists of :class:`EnergyBucket`.
        A bucket that is both certain and covered appears in both lists.
    """
    buckets = bucket_conditions(
        conditions,
        width=width,
        unit=unit,
        origin=origin,
        assume_unit=assume_unit,
    )
    definite = [bucket for bucket in buckets if bucket.definite]
    possible = [bucket for bucket in buckets if bucket.possible]
    return definite, possible


def energy_buckets(conditions, width=1.0, unit=None, origin=0.0, assume_unit=None):
    """Alias for :func:`bucket_conditions`."""
    return bucket_conditions(
        conditions,
        width=width,
        unit=unit,
        origin=origin,
        assume_unit=assume_unit,
    )


def bin_conditions(conditions, width=1.0, unit=None, origin=0.0, assume_unit=None):
    """Alias for :func:`bucket_conditions`."""
    return bucket_conditions(
        conditions,
        width=width,
        unit=unit,
        origin=origin,
        assume_unit=assume_unit,
    )


def bucket_by_energy(conditions, width=1.0, unit=None, origin=0.0, assume_unit=None):
    """Alias for :func:`bucket_conditions`."""
    return bucket_conditions(
        conditions,
        width=width,
        unit=unit,
        origin=origin,
        assume_unit=assume_unit,
    )


def _bucket_index(value, width, origin):
    return math.floor((value - origin) / width)


def _last_bucket_index(high, width, origin):
    """
    Last bucket touched by a condition ending at ``high``.

    Ranges are treated inclusively at both ends: an energy of 40 eV in a
    20-40 eV ramp is part of the acquisition, so the bucket whose boundary is
    40 eV is covered as well.
    """
    return _bucket_index(high, width, origin)


def bucket_spectra(
    spectra, width=1.0, unit=None, origin=0.0, assume_unit=None
):
    """
    Bucket a collection of spectra by their collision energies.

    Arguments:
        spectra (iterable): pymzML ``Spectrum`` objects
        width (float): bucket width
        unit (str or None): target unit
        origin (float): bucket grid alignment
        assume_unit (str or None): unit for vendor values without one

    Returns:
        dict: ``{bucket: {"definite": [spectrum IDs],
        "possible": [spectrum IDs]}}``.  A spectrum appears in both lists
        when it has a fixed energy and a mixture covering the bucket.
    """
    aggregated = {}
    for spectrum in spectra:
        buckets = bucket_conditions(
            spectrum.energy_conditions,
            width=width,
            unit=unit,
            origin=origin,
            assume_unit=assume_unit,
        )
        for bucket in buckets:
            key = (bucket.unit, bucket.start, bucket.end)
            entry = aggregated.setdefault(
                key, {"definite": [], "possible": []}
            )
            if bucket.definite:
                entry["definite"].append(spectrum.ID)
            if bucket.possible:
                entry["possible"].append(spectrum.ID)

    result = {}
    for (bucket_unit, start, end), entry in aggregated.items():
        bucket = EnergyBucket(
            start,
            end,
            unit=bucket_unit,
            definite=bool(entry["definite"]),
            possible=bool(entry["possible"]),
        )
        result[bucket] = entry
    return result


def parse_text(value, unit=None):
    """
    Parse a textual range or number into an :class:`AcquisitionEnergy`.

    Accepted forms include ``"35"``, ``"35 eV"``, ``"25-40"``,
    ``"25-40 eV"``, ``"30%"``, ``"25,30,35"`` and
    ``"25 - 40 electronvolt"``.  Hyphen and ``to`` ranges are continuous;
    comma/semicolon lists are stepped mixtures.
    """
    if value is None:
        raise ValueError("Cannot parse empty energy value")
    text = str(value).strip()
    if text == "":
        raise ValueError("Cannot parse empty energy value")

    if text.endswith("%"):
        return AcquisitionEnergy(float(text[:-1].strip()), unit=PERCENT)

    parts = text.split()
    parsed_unit = unit
    if parts:
        last = parts[-1].lower()
        if last in UNIT_NAMES:
            parsed_unit = UNIT_NAMES[last]
            parts = parts[:-1]
    numeric = " ".join(parts).replace("–", "-")
    if parsed_unit is None:
        parsed_unit = EV

    range_match = _RANGE_PATTERN.match(numeric)
    if range_match:
        return AcquisitionEnergy(
            float(range_match.group(1)),
            float(range_match.group(2)),
            unit=parsed_unit,
        )

    values = _split_values(numeric)
    if len(values) > 1:
        return AcquisitionEnergy(
            min(values), max(values), unit=parsed_unit, steps=values
        )
    if not values:
        raise ValueError("Cannot parse energy value: {0!r}".format(value))
    return AcquisitionEnergy(values[0], unit=parsed_unit)
