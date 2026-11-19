# -*- coding: utf-8 -*-
"""
RDF term identity for cross-shard summaries.

Every RDF term that appears in a shard summary is reduced to a canonical
*identity string*. Two terms occurring in different shards (or in different
versions of the same shard) are considered the same RDF node if and only if
their identity strings are equal. The identity rules mirror exactly how
rdflib collapses terms when triples are unioned into a single graph:

* ``URIRef``      -> the IRI string.
* ``Literal``     -> the (normalised) lexical form, the lower-cased language
  tag (or ``None``) and the datatype IRI (or ``None``). rdflib already
  canonicalises lexical forms of recognised datatypes at construction time
  (e.g. ``"01"^^xsd:integer`` becomes ``"1"^^xsd:integer``) and compares
  language tags case-insensitively, so the tuple
  ``(str(term), lang.lower() or None, datatype or None)`` is a precise
  term identity.
* ``BNode``       -> blank nodes have no graph-global identity of their own,
  so the caller-supplied *blank node convention* maps a
  ``(source_id, local label)`` pair to a global identity. The default
  convention scopes every blank node to its source shard (blank nodes from
  different shards are never equal), which matches the behaviour of naively
  merging independently parsed documents. A deployment that skolemises or
  otherwise coordinates blank node labels across shards can inject its own
  convention.

Summaries only carry the SHA-256 *digest* of an identity string, never the
term itself, so a normal validation round does not move source data. The
actual terms are fetched from the originating shard only when a violation
needs reviewable evidence.
"""

import hashlib
import json

from typing import Callable, Optional, Union

from rdflib import BNode, Literal, URIRef


# A blank node convention maps (source_id, local_bnode_label) -> global identity string.
# ``source_id`` is the shard identifier, or ``None`` for terms that originate
# in the shapes graph (e.g. a blank node used in sh:targetNode).
BNodeConvention = Callable[[Optional[str], str], str]

#: Source id used for terms that originate in the shapes graph rather than a data shard.
SHAPES_GRAPH_SOURCE: Optional[str] = None


def default_bnode_convention(source_id: Optional[str], label: str) -> str:
    """Default blank node convention: blank nodes are scoped to their source."""
    return "{}:{}".format(source_id, label)


def term_identity(term: Union[URIRef, BNode, Literal], source_id: Optional[str], convention: BNodeConvention) -> str:
    """Compute the canonical cross-shard identity string of an RDF term."""
    if isinstance(term, URIRef):
        return "U:" + str(term)
    if isinstance(term, BNode):
        return "B:" + convention(source_id, str(term))
    if isinstance(term, Literal):
        lang = term.language
        datatype = term.datatype
        return "L:" + json.dumps(
            [
                str(term),
                str(lang).lower() if lang else None,
                str(datatype) if datatype is not None else None,
            ],
            ensure_ascii=False,
        )
    raise TypeError("Cannot compute a cross-shard identity for {!r}".format(term))


def identity_digest(identity: str) -> str:
    """Compact, summary-safe representation of a term identity."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def term_digest(term: Union[URIRef, BNode, Literal], source_id: Optional[str], convention: BNodeConvention) -> str:
    """Digest of the identity of ``term``; this is what summaries carry."""
    return identity_digest(term_identity(term, source_id, convention))


def literal_lang(term) -> Optional[str]:
    """The (lower-cased) language tag of a literal, or ``None``.

    Language tags compare case-insensitively in RDF 1.1, so the lower-cased
    tag is the canonical form used by sh:uniqueLang evaluation.
    """
    if isinstance(term, Literal) and term.language:
        return str(term.language).lower()
    return None


def term_from_identity(identity: str) -> Optional[Union[URIRef, Literal]]:
    """Reconstruct a term from its identity string.

    Only works for identities that are self-contained (IRIs and literals).
    Blank node identities are source-relative: the actual term must be
    fetched from the shard that produced it, so this returns ``None``.
    """
    if identity.startswith("U:"):
        return URIRef(identity[2:])
    if identity.startswith("L:"):
        lexical, lang, datatype = json.loads(identity[2:])
        return Literal(lexical, lang=lang, datatype=datatype)
    return None
