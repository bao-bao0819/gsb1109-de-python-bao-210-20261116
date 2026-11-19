# -*- coding: utf-8 -*-
"""
Shard-side protocol for cross-shard validation.

A *shard* holds one version of one slice of the entity data. During normal
validation a shard only ever emits compact summaries:

* the ``rdfs:subClassOf`` pairs it holds (needed to reproduce pySHACL's
  in-data-graph subclass expansion of ``sh:targetClass``),
* the focus-node identities it can resolve for the requested targets,
* per (predicate, focus) the *digests* of the value identities plus, for
  language-tagged strings, the language tag.

No complete source graph and no raw value ever leaves the shard during a
validation round. Only when the coordinator has found a violation does it
call :meth:`ShardProtocol.fetch_values` / :meth:`ShardProtocol.find_term` to
pull the actual, reviewable terms from the originating shard.

:class:`LocalShard` implements the protocol against an in-memory rdflib
graph. A remote deployment can implement the same protocol over a network
boundary; the coordinator only relies on the protocol.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

import rdflib

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS
from rdflib.term import Node

from pyshacl.sharded.identity import (
    BNodeConvention,
    default_bnode_convention,
    literal_lang,
    term_digest,
    term_from_identity,
    term_identity,
)

RDF_type = RDF.type
RDFS_subClassOf = RDFS.subClassOf

#: (digest, lower-cased language tag or None)
ValueRecord = Tuple[str, Optional[str]]

#: predicate IRI -> focus identity -> value records
ValueSummary = Dict[str, Dict[str, Set[ValueRecord]]]


class TargetSummary:
    """Focus-node identities a shard resolves for one target request."""

    __slots__ = ('members', 'subjects', 'objects')

    def __init__(self) -> None:
        #: class identity -> focus identities with a local rdf:type triple
        self.members: Dict[str, Set[str]] = {}
        #: predicate IRI -> subject identities
        self.subjects: Dict[str, Set[str]] = {}
        #: predicate IRI -> object identities
        self.objects: Dict[str, Set[str]] = {}


class ShardProtocol:
    """The interface the coordinator uses to talk to one shard."""

    shard_id: str
    version: str

    def subclass_pairs(self) -> Set[Tuple[str, str]]:
        """Identities of all local ``(subClass, superClass)`` pairs."""
        raise NotImplementedError

    def target_summary(
        self,
        class_identities: Set[str],
        subjects_of_predicates: Set[str],
        objects_of_predicates: Set[str],
    ) -> TargetSummary:
        """Focus identities for the requested classes and predicates."""
        raise NotImplementedError

    def value_summary(self, queries: Dict[str, Set[str]]) -> ValueSummary:
        """Value digests for the requested (predicate -> focus identities)."""
        raise NotImplementedError

    def fetch_values(self, predicate: str, focus_identity: str, digests: Set[str]) -> List[Node]:
        """Actual value terms for ``(focus, predicate)`` matching ``digests``.

        Only called when a violation needs reviewable evidence.
        """
        raise NotImplementedError

    def find_term(self, identity: str) -> Optional[Node]:
        """The local term for an identity, if this shard holds it."""
        raise NotImplementedError

    def namespaces(self) -> Iterable[Tuple[str, str]]:
        """Namespace prefix bindings known to this shard (for messages)."""
        raise NotImplementedError


class LocalShard(ShardProtocol):
    """A shard backed by one immutable version of an rdflib graph."""

    def __init__(
        self,
        shard_id: str,
        version: str,
        graph: Graph,
        bnode_convention: Optional[BNodeConvention] = None,
    ) -> None:
        if not isinstance(graph, (rdflib.Graph, rdflib.Dataset)):
            raise TypeError("LocalShard requires an rdflib Graph or Dataset.")
        if isinstance(graph, rdflib.Dataset):
            graph.default_union = True
        self.shard_id = str(shard_id)
        self.version = str(version)
        self.graph = graph
        self._convention: BNodeConvention = bnode_convention or default_bnode_convention
        self._class_terms: Optional[Dict[str, Node]] = None

    @classmethod
    def from_data(
        cls,
        shard_id: str,
        version: str,
        data,
        data_format: Optional[str] = None,
        bnode_convention: Optional[BNodeConvention] = None,
    ) -> "LocalShard":
        """Build a shard from an rdflib graph or a serialised (default turtle) document."""
        if isinstance(data, (rdflib.Graph, rdflib.Dataset)):
            graph = data
        elif isinstance(data, (str, bytes)):
            graph = Graph()
            graph.parse(data=data, format=data_format or "turtle")
        else:
            raise TypeError("Shard data must be an rdflib Graph or a serialised RDF string.")
        return cls(shard_id, version, graph, bnode_convention=bnode_convention)

    def _identity(self, term: Node) -> str:
        return term_identity(term, self.shard_id, self._convention)

    def _class_registry(self) -> Dict[str, Node]:
        """Identity -> local term for everything in class position locally."""
        if self._class_terms is None:
            registry: Dict[str, Node] = {}
            for _s, _o in self.graph.subject_objects(RDF_type):
                registry.setdefault(self._identity(_s), _s)
                registry.setdefault(self._identity(_o), _o)
            for _s, _o in self.graph.subject_objects(RDFS_subClassOf):
                registry.setdefault(self._identity(_s), _s)
                registry.setdefault(self._identity(_o), _o)
            self._class_terms = registry
        return self._class_terms

    def namespaces(self) -> Iterable[Tuple[str, str]]:
        return [(p, str(n)) for p, n in self.graph.namespace_manager.namespaces()]

    def subclass_pairs(self) -> Set[Tuple[str, str]]:
        return {
            (self._identity(sub), self._identity(sup))
            for sub, sup in self.graph.subject_objects(RDFS_subClassOf)
        }

    def target_summary(
        self,
        class_identities: Set[str],
        subjects_of_predicates: Set[str],
        objects_of_predicates: Set[str],
    ) -> TargetSummary:
        summary = TargetSummary()
        registry = self._class_registry()
        for class_identity in class_identities:
            term = term_from_identity(class_identity)
            if term is None:
                term = registry.get(class_identity)
            if term is None:
                continue
            members = {self._identity(s) for s in self.graph.subjects(RDF_type, term)}
            if members:
                summary.members[class_identity] = members
        for predicate in subjects_of_predicates:
            found = {self._identity(s) for s, _o in self.graph.subject_objects(URIRef(predicate))}
            if found:
                summary.subjects[predicate] = found
        for predicate in objects_of_predicates:
            found = {self._identity(o) for _s, o in self.graph.subject_objects(URIRef(predicate))}
            if found:
                summary.objects[predicate] = found
        return summary

    def value_summary(self, queries: Dict[str, Set[str]]) -> ValueSummary:
        summary: ValueSummary = {}
        for predicate, focus_identities in queries.items():
            if not focus_identities:
                continue
            per_focus: Dict[str, Set[ValueRecord]] = {}
            for subject, value in self.graph.subject_objects(URIRef(predicate)):
                focus_identity = self._identity(subject)
                if focus_identity not in focus_identities:
                    continue
                record = (term_digest(value, self.shard_id, self._convention), literal_lang(value))
                per_focus.setdefault(focus_identity, set()).add(record)
            if per_focus:
                summary[predicate] = per_focus
        return summary

    def fetch_values(self, predicate: str, focus_identity: str, digests: Set[str]) -> List[Node]:
        values: List[Node] = []
        for subject, value in self.graph.subject_objects(URIRef(predicate)):
            if self._identity(subject) != focus_identity:
                continue
            if term_digest(value, self.shard_id, self._convention) in digests:
                values.append(value)
        return values

    def find_term(self, identity: str) -> Optional[Node]:
        term = term_from_identity(identity)
        if term is not None:
            return term
        for node in self.graph.all_nodes():
            if self._identity(node) == identity:
                return node
        return None
