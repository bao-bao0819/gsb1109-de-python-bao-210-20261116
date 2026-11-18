"""
Build an ingestion plan for the members of one or more SAS XPORT libraries.

An XPORT library can hold several kinds of members -- primary datasets,
detail datasets, and standalone code tables.  The same member name can
appear more than once, because each release of a library may publish a
member with the same name.  A plan therefore identifies every member
*occurrence* with a :class:`MemberRef` that carries the member name, its
ordinal position in the ingestion batch, and the release it belongs to.

Callers describe what they need as *dependencies*.  Each dependency names
a member and, optionally, the members it references.  The plan resolves
those references to concrete member occurrences, records every
occurrence's identity, and groups connected members into publication
units.

A detail member that is incomplete -- explicitly flagged, missing key
variables, carrying null key values, or referencing a member that is
absent -- blocks publication of its whole publication unit.  The plan
never presents such a unit as a complete association set.  A complete,
standalone code table remains independently verifiable and acceptable,
even when other units in the same batch are blocked.
"""

# Standard Library
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

# Community Packages
import pandas as pd

# Xport Modules
import xport

LOG = logging.getLogger(__name__)

__all__ = [
    'Dependency',
    'MemberManifest',
    'MemberRef',
    'Plan',
    'PublicationUnit',
    'ROLE_CODE',
    'ROLE_DETAIL',
    'ROLE_PRIMARY',
    'build_plan',
]

ROLE_PRIMARY = 'primary'
ROLE_DETAIL = 'detail'
ROLE_CODE = 'code'
_ROLES = frozenset({ROLE_PRIMARY, ROLE_DETAIL, ROLE_CODE})


@dataclass(frozen=True)
class MemberRef:
    """
    Identity of one member occurrence.

    Member names are not unique across releases, so the ordinal position
    of the occurrence within the ingestion batch is part of its identity.
    """

    name: str
    ordinal: int
    release: str = ''

    def __str__(self):
        """
        Pleasant display value.
        """
        if self.release:
            return f'{self.name}#{self.ordinal}({self.release})'
        return f'{self.name}#{self.ordinal}'


@dataclass(frozen=True)
class Dependency:
    """
    A caller-requested member and the members it references.

    ``release`` and ``ordinal`` are optional disambiguators for cases
    where the same member name occurs in more than one release.
    """

    name: str
    requires: tuple = ()
    release: Optional[str] = None
    ordinal: Optional[int] = None

    @classmethod
    def from_value(cls, value):
        """
        Coerce a string, tuple, mapping, or ``Dependency`` into one.
        """
        if isinstance(value, Dependency):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, tuple):
            return cls(*value)
        if isinstance(value, Mapping):
            return cls(
                name=value['name'],
                requires=tuple(value.get('requires', ())),
                release=value.get('release'),
                ordinal=value.get('ordinal'),
            )
        raise TypeError(f'Cannot interpret dependency {value!r}')


@dataclass(frozen=True)
class MemberManifest:
    """
    Planned disposition of a single member occurrence.
    """

    ref: MemberRef
    role: str
    references: tuple = ()
    unresolved: tuple = ()
    issues: tuple = ()

    @property
    def complete(self):
        """
        Whether the member itself passed every integrity check.
        """
        return not self.issues


@dataclass(frozen=True)
class PublicationUnit:
    """
    A connected set of members that must be published together.
    """

    members: tuple
    issues: tuple = ()

    @property
    def ready(self):
        """
        Whether the unit is complete and may be published.
        """
        return not self.issues

    @property
    def status(self):
        """
        Human-readable publication status.
        """
        return 'ready' if self.ready else 'incomplete'


@dataclass(frozen=True)
class Plan:
    """
    An ingestion plan: member manifests, publication units, and checks.
    """

    manifests: tuple = ()
    units: tuple = ()
    issues: tuple = ()
    _index: dict = field(default_factory=dict, repr=False, compare=False)
    _units_by_ref: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        """
        Build lookup indexes.
        """
        index = {manifest.ref: manifest for manifest in self.manifests}
        units_by_ref = {}
        for unit in self.units:
            for ref in unit.members:
                units_by_ref[ref] = unit
        object.__setattr__(self, '_index', index)
        object.__setattr__(self, '_units_by_ref', units_by_ref)

    def find(self, name, release=None):
        """
        Find every occurrence of a member, optionally filtered by release.
        """
        refs = [ref for ref in self._index if ref.name == name]
        if release is not None:
            refs = [ref for ref in refs if ref.release == release]
        return tuple(refs)

    def get(self, ref):
        """
        Get the manifest for a member reference.
        """
        return self._index[ref]

    def unit_for(self, ref):
        """
        Get the publication unit that contains a member, if any.
        """
        return self._units_by_ref.get(ref)

    def verify(self, ref):
        """
        Verify that a member may be accepted, raising ``ValueError`` if not.

        A member in a publication unit is only verifiable when the whole
        unit is ready, so an incomplete detail blocks the related primary
        member as well.  A standalone member passes when its own manifest
        is clean, regardless of the state of other units.
        """
        manifest = self.get(ref)
        reasons = list(manifest.issues)
        unit = self.unit_for(ref)
        if unit is not None and not unit.ready:
            reasons.extend(issue for issue in unit.issues if issue not in reasons)
        if reasons:
            raise ValueError(f'{ref} cannot be accepted: ' + '; '.join(reasons))
        return None

    @property
    def ready_units(self):
        """
        Publication units that are complete and may be published.
        """
        return tuple(unit for unit in self.units if unit.ready)

    @property
    def blocked_units(self):
        """
        Publication units that must not be published, with their reasons.
        """
        return tuple(unit for unit in self.units if not unit.ready)

    @property
    def acceptable(self):
        """
        Every member occurrence that can currently be verified and accepted.

        Members of blocked publication units are excluded, but clean
        standalone members -- such as a complete code table -- are included
        even when other units are blocked.
        """
        return tuple(ref for ref in self._index if self._is_acceptable(ref))

    def _is_acceptable(self, ref):
        try:
            self.verify(ref)
        except ValueError:
            return False
        return True


def build_plan(
    library,
    dependencies=(),
    *,
    roles=None,
    keys=None,
    incomplete=None,
    releases=None,
):
    """
    Build an ingestion :class:`Plan` from one or more XPORT libraries.

    ``library`` may be a single ``xport.Library``/mapping of datasets, an
    iterable of datasets, or an iterable of ``(release, library)`` tuples.
    The last form allows the same member name to occur in more than one
    release; each dataset gets a unique ordinal across the whole batch.

    ``dependencies`` is an iterable (or mapping) of :class:`Dependency`
    specifications.  Plain member names, ``(name, requires)`` tuples, and
    attribute mappings are all accepted.

    ``roles`` maps member names or ``(name, ordinal)`` pairs to one of
    :data:`ROLE_PRIMARY`, :data:`ROLE_DETAIL`, or :data:`ROLE_CODE`.
    Members that reference others default to the detail role; everyone
    else defaults to the primary role.

    ``keys`` maps member names to their key variable names.  Missing key
    variables or null key values mark a member as incomplete.

    ``incomplete`` explicitly flags member occurrences (by name,
    ``(name, ordinal)`` pair, or :class:`MemberRef`) as incomplete.

    ``releases`` overrides release labels, keyed by ordinal or
    ``(name, ordinal)`` pair.
    """
    roles = roles or {}
    keys = keys or {}
    releases = releases or {}
    incomplete_specs = _coerce_ref_specs(incomplete or ())
    dependency_list = _coerce_dependencies(dependencies)

    occurrences = list(_iter_occurrences(library))
    refs = [
        MemberRef(
            name=getattr(dataset, 'name', None),
            ordinal=ordinal,
            release=_release_label(dataset, source_release, ordinal, releases),
        )
        for ordinal, (source_release, dataset) in enumerate(occurrences)
    ]
    datasets = {ref.ordinal: dataset for ref, (_, dataset) in zip(refs, occurrences)}

    plan_issues = []
    requested = set()
    dependency_by_ordinal = defaultdict(list)
    for dependency in dependency_list:
        matches = _resolve(refs, dependency.name, dependency.release, dependency.ordinal)
        if not matches:
            plan_issues.append(
                f"dependency {dependency.name!r} does not match any member"
                + (f' of release {dependency.release!r}' if dependency.release else '')
            )
            continue
        if len(matches) > 1:
            plan_issues.append(
                f"dependency {dependency.name!r} matches more than one release; "
                'specify a release or ordinal'
            )
            for ref in matches:
                dependency_by_ordinal[ref.ordinal].append((dependency, True))
                requested.add(ref)
            continue
        dependency_by_ordinal[matches[0].ordinal].append((dependency, False))
        requested.add(matches[0])

    manifests = []
    for ref in refs:
        dataset = datasets[ref.ordinal]
        matched = dependency_by_ordinal.get(ref.ordinal, [])
        references, unresolved, dep_issues = _resolve_references(ref, refs, matched)
        issues = list(dep_issues)
        issues.extend(_check_integrity(ref, dataset, keys))
        if _is_flagged_incomplete(ref, incomplete_specs):
            issues.append(f'{ref} was explicitly flagged incomplete')
        role = _role_for(ref, roles, bool(references))
        manifests.append(
            MemberManifest(
                ref=ref,
                role=role,
                references=tuple(references),
                unresolved=tuple(unresolved),
                issues=tuple(issues),
            )
        )

    manifest_by_ref = {manifest.ref: manifest for manifest in manifests}
    units = _build_units(requested, manifest_by_ref)
    return Plan(manifests=tuple(manifests), units=tuple(units), issues=tuple(plan_issues))


def _iter_occurrences(library):
    """
    Yield ``(source_release, dataset)`` pairs in ingestion order.
    """
    if isinstance(library, pd.DataFrame):
        yield '', library
        return

    if isinstance(library, xport.Library):
        for dataset in library.values():
            yield getattr(library, 'sas_version', ''), dataset
        return

    if isinstance(library, Mapping):
        for dataset in library.values():
            yield '', dataset
        return

    for item in library:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Mapping):
            source_release, source_library = item
            for dataset in source_library.values():
                yield source_release, dataset
        else:
            yield '', item


def _release_label(dataset, source_release, ordinal, overrides):
    """
    Determine the release label for one occurrence.
    """
    name = getattr(dataset, 'name', None)
    if (name, ordinal) in overrides:
        return str(overrides[(name, ordinal)])
    if ordinal in overrides:
        return str(overrides[ordinal])
    if source_release:
        return str(source_release)
    return getattr(dataset, 'sas_version', None) or ''


def _coerce_dependencies(raw):
    """
    Coerce the dependencies argument into a list of ``Dependency``.
    """
    if isinstance(raw, Mapping):
        dependencies = []
        for name, spec in raw.items():
            if spec is None:
                dependencies.append(Dependency(name=name))
            elif isinstance(spec, Mapping):
                dependencies.append(
                    Dependency(
                        name=name,
                        requires=tuple(spec.get('requires', ())),
                        release=spec.get('release'),
                        ordinal=spec.get('ordinal'),
                    )
                )
            else:
                dependencies.append(Dependency(name=name, requires=tuple(spec)))
        return dependencies
    return [Dependency.from_value(value) for value in raw]


def _coerce_ref_specs(values):
    """
    Coerce member references given as names, pairs, or ``MemberRef``.
    """
    specs = []
    for value in values:
        if isinstance(value, MemberRef):
            specs.append((value.name, value.ordinal))
        elif isinstance(value, tuple):
            name, ordinal = value
            specs.append((name, ordinal))
        else:
            specs.append((value, None))
    return specs


def _resolve(refs, name, release=None, ordinal=None):
    """
    Resolve a member name to matching occurrences.
    """
    matches = [ref for ref in refs if ref.name == name]
    if ordinal is not None:
        matches = [ref for ref in matches if ref.ordinal == ordinal]
    if release is not None:
        matches = [ref for ref in matches if ref.release == release]
    return matches


def _resolve_references(ref, refs, matched):
    """
    Resolve the ``requires`` lists of matching dependencies.
    """
    references = []
    unresolved = []
    issues = []
    for dependency, ambiguous in matched:
        if ambiguous:
            issues.append(
                f'{ref} shares its name with another release; '
                'specify a release or ordinal in the dependency'
            )
        for spec in dependency.requires:
            required_name, release, ordinal = _unpack_reference(spec)
            targets = _resolve_reference(refs, ref, required_name, release, ordinal)
            if len(targets) == 1:
                target = targets[0]
                if target not in references:
                    references.append(target)
            elif not targets:
                if required_name not in unresolved:
                    unresolved.append(required_name)
                issues.append(f'{ref} references missing member {required_name!r}')
            else:
                issues.append(
                    f'{ref} references ambiguous member name {required_name!r}; '
                    'specify a release or ordinal'
                )
    return references, unresolved, issues


def _unpack_reference(spec):
    """
    Split a reference specification into ``(name, release, ordinal)``.
    """
    if isinstance(spec, Dependency):
        return spec.name, spec.release, spec.ordinal
    if isinstance(spec, Mapping):
        return spec['name'], spec.get('release'), spec.get('ordinal')
    if isinstance(spec, tuple):
        name = spec[0]
        release = spec[1] if len(spec) > 1 else None
        ordinal = spec[2] if len(spec) > 2 else None
        return name, release, ordinal
    return spec, None, None


def _resolve_reference(refs, source, name, release, ordinal):
    """
    Resolve a referenced name, preferring the same release as the source.

    A reference may pin the target's release or ordinal.  Otherwise an
    occurrence in the referencing member's own release wins; ambiguity
    remains only when that release itself contains duplicates.
    """
    targets = _resolve(refs, name, release=release, ordinal=ordinal)
    if release is None and ordinal is None:
        same_release = [target for target in targets if target.release == source.release]
        if len(same_release) == 1:
            return same_release
    return targets


def _check_integrity(ref, dataset, key_map):
    """
    Check a member occurrence for structural incompleteness.
    """
    issues = []
    columns = list(getattr(dataset, 'columns', []))
    if not columns:
        issues.append(f'{ref} has no variables')
    for column in key_map.get(ref.name, ()):
        if column not in columns:
            issues.append(f'{ref} is missing key variable {column!r}')
            continue
        null_count = int(dataset[column].isna().sum())
        if null_count:
            issues.append(
                f'{ref} has {null_count} missing key value(s) in {column!r}'
            )
    return issues


def _is_flagged_incomplete(ref, incomplete_specs):
    """
    Whether an occurrence was explicitly flagged incomplete.
    """
    for name, ordinal in incomplete_specs:
        if ref.name == name and (ordinal is None or ordinal == ref.ordinal):
            return True
    return False


def _role_for(ref, roles, has_references):
    """
    Determine the member role, honoring caller overrides.
    """
    role = roles.get((ref.name, ref.ordinal), roles.get(ref.name))
    if role is None:
        role = ROLE_DETAIL if has_references else ROLE_PRIMARY
    if role not in _ROLES:
        raise ValueError(f'Unknown role {role!r} for {ref}')
    return role


def _build_units(roots, manifest_by_ref):
    """
    Group connected requested members into publication units.
    """
    units = []
    visited = set()
    for root in roots:
        if root in visited:
            continue
        members = []
        stack = [root]
        while stack:
            ref = stack.pop()
            if ref in visited:
                continue
            visited.add(ref)
            manifest = manifest_by_ref.get(ref)
            if manifest is None:
                continue
            members.append(ref)
            for target in manifest.references:
                if target not in visited:
                    stack.append(target)
        members.sort(key=lambda member: member.ordinal)
        issues = []
        for ref in members:
            issues.extend(manifest_by_ref[ref].issues)
        units.append(
            PublicationUnit(
                members=tuple(members),
                issues=tuple(dict.fromkeys(issues)),
            )
        )
    units.sort(key=lambda unit: unit.members[0].ordinal)
    return units
