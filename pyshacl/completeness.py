# -*- coding: utf-8 -*-
"""
Partial (open-world) SHACL validation with completeness declarations.

Standard SHACL validation is closed-world: whatever is absent from the data
graph is treated as violated or satisfied based only on the triples at hand.
When only a slice of the entities is exchanged, the relations that are
missing from the received graph may simply live in graphs that were not
shipped. This module makes that distinction explicit.

A :class:`CompletenessDeclaration` states which scopes of the input are
complete (closed). Closedness only ever applies to the declared scopes;
everything else stays open. Based on the declarations, validation results
are separated into three buckets:

* **proven violations** -- violations that no additional data can retract
  (either the constraint is growth-stable, e.g. ``sh:maxCount`` or
  ``sh:datatype``, or every scope the constraint depends on is declared
  complete),
* **proven conformances** -- constraints satisfied under the declared
  completeness, and
* **unknowns** -- outcomes for which evidence is insufficient. Every
  unknown carries the *missing scope* (focus node x predicate) that a
  query would still need in order to become decidable.

Because proven results are final, refinement is incremental: after
supplementing the graph with another slice of data (and possibly widening
the declarations), only the previous unknowns are re-evaluated and can turn
into definite conclusions under the new evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

from rdflib import Graph, Literal, URIRef
from rdflib.collection import Collection

from .consts import RDF, RDFS, SH
from .entrypoints import validate as _shacl_validate
from .helper.expression_helper import value_nodes_from_path

__all__ = [
    'CompletenessDeclaration',
    'MissingScope',
    'ProvenViolation',
    'ProvenConformance',
    'UnknownItem',
    'PartialValidationReport',
    'PartialValidator',
    'validate_partial',
]


# Constraint components whose violations cannot be retracted by adding more
# triples to the data graph (the evidence is the *presence* of bad data).
_GROWTH_STABLE_COMPONENTS: FrozenSet[URIRef] = frozenset(
    SH[str(local)]
    for local in (
        'MaxCountConstraintComponent',
        'QualifiedMaxCountConstraintComponent',
        'DatatypeConstraintComponent',
        'NodeKindConstraintComponent',
        'MinLengthConstraintComponent',
        'MaxLengthConstraintComponent',
        'MinExclusiveConstraintComponent',
        'MinInclusiveConstraintComponent',
        'MaxExclusiveConstraintComponent',
        'MaxInclusiveConstraintComponent',
        'PatternConstraintComponent',
        'InConstraintComponent',
        'LanguageInConstraintComponent',
        'UniqueLangConstraintComponent',
        'DisjointConstraintComponent',
        'LessThanConstraintComponent',
        'LessThanOrEqualsConstraintComponent',
        'ClosedConstraintComponent',
    )
)

# Components that delegate to member/nested shapes; their verdicts are
# derived recursively from the inner shapes' outcomes.
_NODE_COMPONENT = SH['NodeConstraintComponent']
_PROPERTY_COMPONENT = SH['PropertyConstraintComponent']
_QUALIFIED_VALUE_COMPONENT = SH['QualifiedValueShapeConstraintComponent']
_NOT_COMPONENT = SH['NotConstraintComponent']
_AND_COMPONENT = SH['AndConstraintComponent']
_OR_COMPONENT = SH['OrConstraintComponent']
_XONE_COMPONENT = SH['XoneConstraintComponent']

_COMPOSITE_COMPONENTS: FrozenSet[URIRef] = frozenset(
    (
        _NODE_COMPONENT,
        _PROPERTY_COMPONENT,
        _QUALIFIED_VALUE_COMPONENT,
        _NOT_COMPONENT,
        _AND_COMPONENT,
        _OR_COMPONENT,
        _XONE_COMPONENT,
    )
)

# For sh:and / sh:node / sh:property / sh:qualifiedValueShape a single proven
# inner failure is enough to prove the outer violation. For sh:or / sh:xone /
# sh:not the outer violation is only provable once every dependency is
# complete, so they are handled conservatively.
_CONJUNCTIVE_COMPOSITES: FrozenSet[URIRef] = frozenset(
    (_NODE_COMPONENT, _PROPERTY_COMPONENT, _QUALIFIED_VALUE_COMPONENT, _AND_COMPONENT)
)

# Components that compare against another property of the same focus node.
_PROPERTY_PAIR_COMPONENTS: FrozenSet[URIRef] = frozenset(
    SH[str(local)]
    for local in (
        'EqualsConstraintComponent',
        'DisjointConstraintComponent',
        'LessThanConstraintComponent',
        'LessThanOrEqualsConstraintComponent',
    )
)

_CLOSED_COMPONENT = SH['ClosedConstraintComponent']
_SPARQL_COMPONENT = SH['SPARQLConstraintComponent']

# SHACL parameter -> constraint component, for parameters that may appear
# directly on a node shape.
_NODE_LEVEL_PARAMS: Dict[URIRef, URIRef] = {
    SH['class']: SH['ClassConstraintComponent'],
    SH['datatype']: SH['DatatypeConstraintComponent'],
    SH['nodeKind']: SH['NodeKindConstraintComponent'],
    SH['minExclusive']: SH['MinExclusiveConstraintComponent'],
    SH['minInclusive']: SH['MinInclusiveConstraintComponent'],
    SH['maxExclusive']: SH['MaxExclusiveConstraintComponent'],
    SH['maxInclusive']: SH['MaxInclusiveConstraintComponent'],
    SH['minLength']: SH['MinLengthConstraintComponent'],
    SH['maxLength']: SH['MaxLengthConstraintComponent'],
    SH['pattern']: SH['PatternConstraintComponent'],
    SH['languageIn']: SH['LanguageInConstraintComponent'],
    SH['uniqueLang']: SH['UniqueLangConstraintComponent'],
    SH['equals']: SH['EqualsConstraintComponent'],
    SH['disjoint']: SH['DisjointConstraintComponent'],
    SH['lessThan']: SH['LessThanConstraintComponent'],
    SH['lessThanOrEquals']: SH['LessThanOrEqualsConstraintComponent'],
    SH['not']: _NOT_COMPONENT,
    SH['and']: _AND_COMPONENT,
    SH['or']: _OR_COMPONENT,
    SH['xone']: _XONE_COMPONENT,
    SH['node']: _NODE_COMPONENT,
    SH['closed']: _CLOSED_COMPONENT,
    SH['hasValue']: SH['HasValueConstraintComponent'],
    SH['in']: SH['InConstraintComponent'],
    SH['sparql']: _SPARQL_COMPONENT,
}

# Additional parameters that only take effect on property shapes.
_PROPERTY_ONLY_PARAMS: Dict[URIRef, URIRef] = {
    SH['minCount']: SH['MinCountConstraintComponent'],
    SH['maxCount']: SH['MaxCountConstraintComponent'],
    SH['qualifiedValueShape']: _QUALIFIED_VALUE_COMPONENT,
    SH['qualifiedMinCount']: SH['QualifiedMinCountConstraintComponent'],
    SH['qualifiedMaxCount']: SH['QualifiedMaxCountConstraintComponent'],
}

_PROPERTY_LEVEL_PARAMS: Dict[URIRef, URIRef] = dict(_NODE_LEVEL_PARAMS)
_PROPERTY_LEVEL_PARAMS.update(_PROPERTY_ONLY_PARAMS)


def _as_graph(source: Union[Graph, str, bytes]) -> Graph:
    if isinstance(source, Graph):
        return source
    graph = Graph()
    if isinstance(source, (str, bytes)):
        try:
            graph.parse(data=source, format='turtle')
        except Exception:
            try:
                graph.parse(data=source, format='xml')
            except Exception:
                graph.parse(source)  # treat as file path or URL
        return graph
    raise TypeError("Expected an rdflib.Graph, a serialized graph, or a path/URL.")


def _path_predicates(shapes_graph: Graph, path_node, _depth: int = 0) -> Set[URIRef]:
    """Collect every predicate IRI that occurs inside a sh:path structure."""
    if isinstance(path_node, URIRef):
        return {path_node}
    if _depth > 20:
        return set()
    predicates: Set[URIRef] = set()
    for modifier in (SH['inversePath'], SH['zeroOrMorePath'], SH['oneOrMorePath'], SH['zeroOrOnePath']):
        for sub in shapes_graph.objects(path_node, modifier):
            predicates |= _path_predicates(shapes_graph, sub, _depth + 1)
    for alt_list in shapes_graph.objects(path_node, SH['alternativePath']):
        for member in Collection(shapes_graph, alt_list):
            predicates |= _path_predicates(shapes_graph, member, _depth + 1)
    if (path_node, RDF['first'], None) in shapes_graph:
        for member in Collection(shapes_graph, path_node):
            predicates |= _path_predicates(shapes_graph, member, _depth + 1)
    return predicates


class _ShapesGraphShim:
    """Minimal stand-in exposing .graph for value_nodes_from_path."""

    __slots__ = ('graph',)

    def __init__(self, graph: Graph):
        self.graph = graph


def _sorted_scopes(scopes) -> Tuple['MissingScope', ...]:
    return tuple(sorted(scopes, key=str))


@dataclass(frozen=True)
class _CompletenessEntry:
    subject: Optional[URIRef]
    of_class: Optional[URIRef]
    predicate: Optional[URIRef]


class CompletenessDeclaration:
    """
    Declares which scopes of the input graph are complete (closed-world).

    Each entry covers a (subject, predicate) scope. ``subject=None`` means
    any subject, ``predicate=None`` means any predicate, and ``of_class``
    restricts the entry to instances of a class. Closedness only applies to
    the declared scopes; everything else remains open-world.
    """

    def __init__(self, entries: Optional[Iterable[_CompletenessEntry]] = None):
        self._entries: List[_CompletenessEntry] = list(entries or [])

    def declare(
        self,
        subject: Optional[Union[str, URIRef]] = None,
        of_class: Optional[Union[str, URIRef]] = None,
        predicate: Optional[Union[str, URIRef]] = None,
    ) -> 'CompletenessDeclaration':
        if subject is not None and of_class is not None:
            raise ValueError("Declare either a subject or a class scope, not both.")
        entry = _CompletenessEntry(
            URIRef(subject) if subject is not None else None,
            URIRef(of_class) if of_class is not None else None,
            URIRef(predicate) if predicate is not None else None,
        )
        self._entries.append(entry)
        return self

    @classmethod
    def everything(cls) -> 'CompletenessDeclaration':
        """A declaration marking the whole input graph as complete."""
        return cls([_CompletenessEntry(None, None, None)])

    def merge(self, other: 'CompletenessDeclaration') -> 'CompletenessDeclaration':
        return CompletenessDeclaration(self._entries + list(other.entries))

    @property
    def entries(self) -> Tuple[_CompletenessEntry, ...]:
        return tuple(self._entries)

    def covers(
        self,
        subject: URIRef,
        predicate: Optional[URIRef],
        subject_types: Iterable[URIRef] = (),
    ) -> bool:
        """
        True if (subject, predicate) is declared complete. ``predicate=None``
        asks whether *all* predicates of the subject are covered.
        """
        types = set(subject_types)
        for entry in self._entries:
            if entry.subject is not None and entry.subject != subject:
                continue
            if entry.of_class is not None and entry.of_class not in types:
                continue
            if entry.predicate is not None and entry.predicate != predicate:
                continue
            return True
        return False

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return "CompletenessDeclaration(entries={!r})".format(self._entries)


@dataclass(frozen=True)
class MissingScope:
    """A scope whose data was not received (or not declared complete).

    ``predicate=None`` means the scope spans every predicate of ``focus``.
    """

    focus: URIRef
    predicate: Optional[URIRef]


@dataclass(frozen=True)
class ProvenViolation:
    """A violation that no additional data can retract."""

    focus: URIRef
    shape: URIRef
    component: URIRef
    path: Optional[URIRef]
    value: Optional[object] = None
    severity: Optional[URIRef] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class ProvenConformance:
    """A constraint instance satisfied under the declared completeness."""

    focus: URIRef
    shape: URIRef
    component: URIRef
    path: Optional[URIRef]


@dataclass(frozen=True)
class UnknownItem:
    """An undecidable constraint instance plus the scope needed to decide it.

    ``leans`` records which way the current (incomplete) evidence points:
    ``'violation'`` if a violation was reported but may be retracted by
    incoming data, ``'conformance'`` if no violation was reported but one
    may still appear.
    """

    focus: URIRef
    shape: URIRef
    component: URIRef
    path: Optional[URIRef]
    leans: str
    missing: Tuple[MissingScope, ...]


@dataclass(frozen=True)
class _ResultRecord:
    focus: URIRef
    component: URIRef
    path: Optional[object]
    value: Optional[object]
    severity: Optional[URIRef]
    message: Optional[str]
    source_shape: Optional[URIRef]


@dataclass(frozen=True)
class _ConstraintInstance:
    owner: object  # shape or property-shape node carrying the parameter
    component: URIRef
    param: URIRef
    path: Optional[object]


class PartialValidationReport:
    """Three-valued outcome of validating an entity slice."""

    def __init__(
        self,
        violations: Sequence[ProvenViolation],
        conformances: Sequence[ProvenConformance],
        unknowns: Sequence[UnknownItem],
        focus_nodes: Sequence[URIRef],
    ):
        self.violations: Tuple[ProvenViolation, ...] = tuple(violations)
        self.conformances: Tuple[ProvenConformance, ...] = tuple(conformances)
        self.unknowns: Tuple[UnknownItem, ...] = tuple(unknowns)
        self.focus_nodes: Tuple[URIRef, ...] = tuple(focus_nodes)

    @property
    def conforms(self) -> bool:
        """True if no provable violation was found (unknowns may remain)."""
        return len(self.violations) == 0

    @property
    def decisive(self) -> bool:
        """True if every constraint instance was decidable."""
        return len(self.unknowns) == 0

    def missing_scope(self) -> FrozenSet[MissingScope]:
        """All scopes a query still needs to make every unknown decidable."""
        scope: Set[MissingScope] = set()
        for item in self.unknowns:
            scope |= set(item.missing)
        return frozenset(scope)

    def unknowns_for(self, focus: URIRef) -> Tuple[UnknownItem, ...]:
        return tuple(item for item in self.unknowns if item.focus == focus)

    def summary(self) -> str:
        return (
            "{} proven violation(s), {} proven conformance(s), {} unknown(s)".format(
                len(self.violations), len(self.conformances), len(self.unknowns)
            )
        )

    def __repr__(self) -> str:
        return "PartialValidationReport({})".format(self.summary())


class PartialValidator:
    """
    Incremental three-valued SHACL validator for entity slices.

    Holds the shapes graph, the received data graph and the completeness
    declarations. Data may only grow (``add_data`` merges new slices);
    proven results are cached and carried over, so re-validation after a
    supplement only re-evaluates the previous unknowns.
    """

    def __init__(
        self,
        shacl_graph: Union[Graph, str, bytes],
        data_graph: Optional[Union[Graph, str, bytes]] = None,
        completeness: Optional[CompletenessDeclaration] = None,
        **validate_kwargs,
    ):
        self._shapes = _as_graph(shacl_graph)
        self._shapes_shim = _ShapesGraphShim(self._shapes)
        self._data = Graph()
        self._completeness = completeness if completeness is not None else CompletenessDeclaration()
        self._validate_kwargs = dict(validate_kwargs)
        self._proven: Dict[Tuple[URIRef, object, URIRef, Optional[object]], object] = {}
        self._instance_cache: Dict[object, Tuple[_ConstraintInstance, ...]] = {}
        if data_graph is not None:
            self.add_data(data_graph)

    @property
    def data_graph(self) -> Graph:
        return self._data

    @property
    def completeness(self) -> CompletenessDeclaration:
        return self._completeness

    def add_data(
        self,
        data: Union[Graph, str, bytes],
        completeness: Optional[CompletenessDeclaration] = None,
    ) -> None:
        """Merge another slice of data (and optionally declarations)."""
        for triple in _as_graph(data):
            self._data.add(triple)
        if completeness is not None:
            self._completeness = self._completeness.merge(completeness)

    def add_completeness(self, declaration: CompletenessDeclaration) -> None:
        self._completeness = self._completeness.merge(declaration)

    def validate(
        self, focus_nodes: Optional[Sequence[Union[str, URIRef]]] = None
    ) -> PartialValidationReport:
        """
        Validate the current data graph, restricted to ``focus_nodes`` (the
        entity slice) if given, and classify every constraint instance as a
        proven violation, a proven conformance, or an unknown.

        Note: the slice restriction is applied by this class, not via
        pySHACL's own ``focus_nodes`` option, because that option also
        suppresses nested shape evaluation on nodes outside the slice --
        exactly the reachable evidence we need to classify.
        """
        focus: Optional[List[URIRef]] = None
        if focus_nodes is not None:
            focus = [f if isinstance(f, URIRef) else URIRef(str(f)) for f in focus_nodes]
        _conforms, report_graph, _report_text = _shacl_validate(
            self._data,
            shacl_graph=self._shapes,
            **self._validate_kwargs,
        )
        results_index = self._index_results(report_graph)
        if focus is None:
            focus = sorted(self._all_target_nodes(), key=str)
        violations: List[ProvenViolation] = []
        conformances: List[ProvenConformance] = []
        unknowns: List[UnknownItem] = []
        for f in focus:
            types = self._types_of(f)
            for shape in self._shapes_for(f, types):
                for instance in self._instances_of(shape):
                    key = (f, instance.owner, instance.component, instance.path)
                    cached = self._proven.get(key)
                    if cached is not None:
                        if isinstance(cached, ProvenViolation):
                            violations.append(cached)
                        else:
                            conformances.append(cached)
                        continue
                    outcome = self._classify_instance(f, instance, results_index, types, frozenset())
                    if isinstance(outcome, ProvenViolation):
                        self._proven[key] = outcome
                        violations.append(outcome)
                    elif isinstance(outcome, ProvenConformance):
                        self._proven[key] = outcome
                        conformances.append(outcome)
                    else:
                        unknowns.append(outcome)
        return PartialValidationReport(violations, conformances, unknowns, focus)

    # -- report indexing -------------------------------------------------

    @staticmethod
    def _index_results(report_graph: Graph) -> Dict[Tuple[URIRef, URIRef], List[_ResultRecord]]:
        index: Dict[Tuple[URIRef, URIRef], List[_ResultRecord]] = {}
        for result in report_graph.subjects(RDF['type'], SH['ValidationResult']):
            focus = report_graph.value(result, SH['focusNode'])
            component = report_graph.value(result, SH['sourceConstraintComponent'])
            if focus is None or component is None:
                continue
            message = report_graph.value(result, SH['resultMessage'])
            record = _ResultRecord(
                focus=focus,
                component=component,
                path=report_graph.value(result, SH['resultPath']),
                value=report_graph.value(result, SH['value']),
                severity=report_graph.value(result, SH['resultSeverity']),
                message=str(message) if message is not None else None,
                source_shape=report_graph.value(result, SH['sourceShape']),
            )
            index.setdefault((focus, component), []).append(record)
        return index

    def _matching_results(
        self,
        focus: URIRef,
        instance: _ConstraintInstance,
        results_index: Dict[Tuple[URIRef, URIRef], List[_ResultRecord]],
    ) -> List[_ResultRecord]:
        records = results_index.get((focus, instance.component), [])
        matched = []
        for record in records:
            if instance.path is None:
                if record.path is None:
                    matched.append(record)
            elif isinstance(instance.path, URIRef):
                if record.path == instance.path:
                    matched.append(record)
            else:
                # Complex paths are re-serialized into the report graph, so
                # match loosely on focus + component.
                matched.append(record)
        return matched

    # -- shapes graph introspection --------------------------------------

    def _is_deactivated(self, shape_node) -> bool:
        return (shape_node, SH['deactivated'], Literal(True)) in self._shapes

    def _types_of(self, focus: URIRef) -> Set[URIRef]:
        types = set(self._data.objects(focus, RDF['type']))
        closure = set(types)
        for type_node in types:
            closure |= set(self._data.transitive_objects(type_node, RDFS['subClassOf']))
        return closure

    def _shapes_for(self, focus: URIRef, types: Set[URIRef]) -> Set[object]:
        shapes: Set[object] = set()
        sg = self._shapes
        shapes |= set(sg.subjects(SH['targetNode'], focus))
        for type_node in types:
            shapes |= set(sg.subjects(SH['targetClass'], type_node))
            # Implicit class targets: a class that is itself a shape.
            if self._is_shape_candidate(type_node):
                shapes.add(type_node)
        for shape, predicate in sg.subject_objects(SH['targetSubjectsOf']):
            if (focus, predicate, None) in self._data:
                shapes.add(shape)
        for shape, predicate in sg.subject_objects(SH['targetObjectsOf']):
            if (None, predicate, focus) in self._data:
                shapes.add(shape)
        return {s for s in shapes if not self._is_deactivated(s)}

    def _is_shape_candidate(self, node) -> bool:
        sg = self._shapes
        if (node, RDF['type'], SH['NodeShape']) in sg or (node, RDF['type'], SH['PropertyShape']) in sg:
            return True
        if (node, SH['property'], None) in sg:
            return True
        return any((node, param, None) in sg for param in _PROPERTY_LEVEL_PARAMS)

    def _all_target_nodes(self) -> Set[URIRef]:
        nodes: Set[URIRef] = set()
        sg = self._shapes
        for _shape, target in sg.subject_objects(SH['targetNode']):
            if isinstance(target, URIRef):
                nodes.add(target)
        for _shape, class_node in sg.subject_objects(SH['targetClass']):
            classes = {class_node} | set(sg.transitive_subjects(RDFS['subClassOf'], class_node))
            for cls in classes:
                nodes |= {n for n in self._data.subjects(RDF['type'], cls) if isinstance(n, URIRef)}
        for _shape, predicate in sg.subject_objects(SH['targetSubjectsOf']):
            nodes |= {n for n in self._data.subjects(predicate, None) if isinstance(n, URIRef)}
        for _shape, predicate in sg.subject_objects(SH['targetObjectsOf']):
            nodes |= {n for n in self._data.objects(None, predicate) if isinstance(n, URIRef)}
        for node in set(sg.subjects(RDF['type'], SH['NodeShape'])):
            nodes |= {n for n in self._data.subjects(RDF['type'], node) if isinstance(n, URIRef)}
        return nodes

    def _instances_of(self, shape) -> Tuple[_ConstraintInstance, ...]:
        cached = self._instance_cache.get(shape)
        if cached is not None:
            return cached
        instances: List[_ConstraintInstance] = []
        sg = self._shapes
        for param, component in _NODE_LEVEL_PARAMS.items():
            if (shape, param, None) in sg:
                instances.append(_ConstraintInstance(shape, component, param, None))
        for property_shape in sg.objects(shape, SH['property']):
            if self._is_deactivated(property_shape):
                continue
            instances.extend(self._property_instances(property_shape))
        result = tuple(instances)
        self._instance_cache[shape] = result
        return result

    def _property_instances(self, property_shape) -> List[_ConstraintInstance]:
        sg = self._shapes
        path = sg.value(property_shape, SH['path'])
        instances: List[_ConstraintInstance] = []
        has_nested_property = (property_shape, SH['property'], None) in sg
        for param, component in _PROPERTY_LEVEL_PARAMS.items():
            if param in (SH['qualifiedMinCount'], SH['qualifiedMaxCount']):
                if (property_shape, SH['qualifiedValueShape'], None) not in sg:
                    continue
            if (property_shape, param, None) in sg:
                instances.append(_ConstraintInstance(property_shape, component, param, path))
        if has_nested_property:
            instances.append(
                _ConstraintInstance(property_shape, _PROPERTY_COMPONENT, SH['property'], path)
            )
            for child in sg.objects(property_shape, SH['property']):
                if not self._is_deactivated(child):
                    instances.extend(self._property_instances(child))
        return instances

    # -- classification ----------------------------------------------------

    def _instance_missing(
        self, focus: URIRef, instance: _ConstraintInstance, types: Set[URIRef]
    ) -> Set[MissingScope]:
        sg = self._shapes
        dependencies: Set[Optional[URIRef]] = set()
        if instance.path is not None:
            dependencies |= _path_predicates(sg, instance.path)
        if instance.component in _PROPERTY_PAIR_COMPONENTS:
            for other in sg.objects(instance.owner, instance.param):
                if isinstance(other, URIRef):
                    dependencies.add(other)
        if instance.component in (_CLOSED_COMPONENT, _SPARQL_COMPONENT):
            dependencies.add(None)
        missing: Set[MissingScope] = set()
        for dependency in dependencies:
            if not self._completeness.covers(focus, dependency, types):
                missing.add(MissingScope(focus, dependency))
        return missing

    def _composite_members(self, instance: _ConstraintInstance) -> List[object]:
        sg = self._shapes
        component = instance.component
        if component in (_NODE_COMPONENT, _QUALIFIED_VALUE_COMPONENT):
            return list(sg.objects(instance.owner, instance.param))
        if component == _PROPERTY_COMPONENT:
            return list(sg.objects(instance.owner, SH['property']))
        if component == _NOT_COMPONENT:
            return list(sg.objects(instance.owner, SH['not']))
        # sh:and / sh:or / sh:xone hold RDF lists of shapes.
        members: List[object] = []
        for shape_list in sg.objects(instance.owner, instance.param):
            members.extend(Collection(sg, shape_list))
        return members

    def _classify_instance(
        self,
        focus: URIRef,
        instance: _ConstraintInstance,
        results_index: Dict[Tuple[URIRef, URIRef], List[_ResultRecord]],
        types: Set[URIRef],
        stack: FrozenSet[Tuple[URIRef, object]],
    ):
        reported = self._matching_results(focus, instance, results_index)
        if instance.component in _COMPOSITE_COMPONENTS:
            return self._classify_composite(focus, instance, reported, results_index, types, stack)
        missing = self._instance_missing(focus, instance, types)
        path = instance.path if isinstance(instance.path, URIRef) else None
        if reported:
            record = reported[0]
            if instance.component in _GROWTH_STABLE_COMPONENTS or not missing:
                return ProvenViolation(
                    focus=focus,
                    shape=record.source_shape or instance.owner,
                    component=instance.component,
                    path=path,
                    value=record.value,
                    severity=record.severity,
                    message=record.message,
                )
            return UnknownItem(
                focus, instance.owner, instance.component, path, 'violation', _sorted_scopes(missing)
            )
        if not missing:
            return ProvenConformance(focus, instance.owner, instance.component, path)
        return UnknownItem(
            focus, instance.owner, instance.component, path, 'conformance', _sorted_scopes(missing)
        )

    def _classify_composite(
        self,
        focus: URIRef,
        instance: _ConstraintInstance,
        reported: List[_ResultRecord],
        results_index: Dict[Tuple[URIRef, URIRef], List[_ResultRecord]],
        types: Set[URIRef],
        stack: FrozenSet[Tuple[URIRef, object]],
    ):
        sg = self._shapes
        members = self._composite_members(instance)
        if instance.path is None:
            values = {focus}
        else:
            values = value_nodes_from_path(self._shapes_shim, focus, instance.path, self._data)
        missing: Set[MissingScope] = set()
        if instance.path is not None:
            for predicate in _path_predicates(sg, instance.path):
                if not self._completeness.covers(focus, predicate, types):
                    missing.add(MissingScope(focus, predicate))
        inner_violation = False
        for value in values:
            for member in members:
                verdict = self._shape_verdict(value, member, results_index, stack)
                missing |= verdict[1]
                inner_violation = inner_violation or verdict[0]
        path = instance.path if isinstance(instance.path, URIRef) else None
        if reported:
            record = reported[0]
            provable = (
                instance.component in _CONJUNCTIVE_COMPOSITES and inner_violation
            ) or not missing
            if provable:
                return ProvenViolation(
                    focus=focus,
                    shape=record.source_shape or instance.owner,
                    component=instance.component,
                    path=path,
                    value=record.value,
                    severity=record.severity,
                    message=record.message,
                )
            return UnknownItem(
                focus, instance.owner, instance.component, path, 'violation', _sorted_scopes(missing)
            )
        if not missing:
            return ProvenConformance(focus, instance.owner, instance.component, path)
        return UnknownItem(
            focus, instance.owner, instance.component, path, 'conformance', _sorted_scopes(missing)
        )

    def _shape_verdict(
        self,
        focus: URIRef,
        shape,
        results_index: Dict[Tuple[URIRef, URIRef], List[_ResultRecord]],
        stack: FrozenSet[Tuple[URIRef, object]],
    ) -> Tuple[bool, Set[MissingScope]]:
        """Aggregate verdict of a nested shape: (has_proven_violation, missing)."""
        if (focus, shape) in stack:
            # Recursive shape reference: stay conservative.
            return False, {MissingScope(focus, None)}
        if self._is_deactivated(shape):
            return False, set()
        next_stack = stack | {(focus, shape)}
        types = self._types_of(focus)
        has_violation = False
        missing: Set[MissingScope] = set()
        for instance in self._instances_of(shape):
            outcome = self._classify_instance(focus, instance, results_index, types, next_stack)
            if isinstance(outcome, ProvenViolation):
                has_violation = True
            elif isinstance(outcome, UnknownItem):
                missing |= set(outcome.missing)
        return has_violation, missing


def validate_partial(
    data_graph: Union[Graph, str, bytes],
    shacl_graph: Union[Graph, str, bytes],
    focus_nodes: Optional[Sequence[Union[str, URIRef]]] = None,
    completeness: Optional[CompletenessDeclaration] = None,
    **validate_kwargs,
) -> PartialValidationReport:
    """
    One-shot three-valued validation of an entity slice.

    Validates only ``focus_nodes`` (or every target if omitted) and splits
    the outcome into proven violations, proven conformances and unknowns
    according to the completeness declarations.
    """
    validator = PartialValidator(
        shacl_graph, data_graph=data_graph, completeness=completeness, **validate_kwargs
    )
    return validator.validate(focus_nodes=focus_nodes)
