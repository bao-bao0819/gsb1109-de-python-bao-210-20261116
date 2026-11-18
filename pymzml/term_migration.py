# -*- coding: utf-8 -*-
# encoding: utf-8
"""
Versioned terminology migration and unified term views.

Terminology vocabularies (psi-ms obo) evolve: accessions are renamed and
old terms get deprecated. This module provides

* :py:class:`TermMigration` - a versioned mapping from original accessions
  to their unified (target) terms, and
* :py:func:`build_unified_view` - a deterministic, offline view over the
  terms declared in a source file.

Every record of the unified view keeps the original accession, the original
label as written in the source file and the version of the migration mapping
that was applied, so re-importing an old partition with the same pinned obo
version and the same mapping version reproduces the identical view. Unknown
terms are kept as-is and are never resolved against a newer online release.
"""

import json
import os


class TermMigration(object):
    """
    Versioned migration mapping between terminology (obo) releases.

    Keyword Arguments:
        terms (dict): mapping of original accession to the unified target
            term, e.g.::

                {
                    "MS:1000127": {
                        "accession": "MS:1000127",
                        "name": "centroid spectrum",
                        "status": "renamed",
                    },
                    "MS:1000001": {
                        "accession": None,
                        "name": None,
                        "status": "deprecated",
                    },
                }

            ``status`` defaults to ``"renamed"``; ``accession`` defaults to
            the original accession.
        version (str): version identifier of the mapping itself. It is
            carried into every record of the unified view.
    """

    def __init__(self, terms=None, version=None):
        self.version = version
        self.terms = {}
        for accession, entry in (terms or {}).items():
            self.terms[accession] = {
                "accession": entry.get("accession", accession),
                "name": entry.get("name"),
                "status": entry.get("status", "renamed"),
            }

    @classmethod
    def from_json(cls, path):
        """
        Load a migration mapping from a JSON file of the form::

            {
                "version": "1.0.0",
                "terms": {"MS:1000127": {"name": "centroid spectrum", ...}}
            }

        Arguments:
            path (str): path to the JSON mapping file

        Returns:
            TermMigration: the loaded mapping
        """
        with open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(terms=payload.get("terms"), version=payload.get("version"))

    @classmethod
    def load(cls, source):
        """
        Normalize a mapping specification into a TermMigration.

        Arguments:
            source: None, a TermMigration, a path to a JSON mapping file or
                a dict (either ``{"version": ..., "terms": {...}}`` or a
                plain accession -> target mapping)

        Returns:
            TermMigration or None
        """
        if source is None or isinstance(source, cls):
            return source
        if isinstance(source, (str, os.PathLike)):
            return cls.from_json(source)
        if isinstance(source, dict):
            if "terms" in source or "version" in source:
                return cls(
                    terms=source.get("terms"), version=source.get("version")
                )
            return cls(terms=source)
        raise TypeError(
            "Cannot build TermMigration from {0!r}".format(type(source))
        )

    def lookup(self, accession):
        """
        Return the migration entry for an original accession.

        Arguments:
            accession (str): original accession as declared in the source file

        Returns:
            dict or None: mapping entry with keys ``accession``, ``name``
                and ``status``
        """
        return self.terms.get(accession)

    def __len__(self):
        return len(self.terms)


def build_unified_view(terms, obo_translator=None, migration=None):
    """
    Generate a deterministic unified terminology view.

    Terms are resolved strictly offline: the migration mapping is consulted
    first, then the obo translator pinned to the version declared by the
    source file. Terms unknown to both keep their original accession and
    label (``status == "unknown"``) and are never replaced by an online
    lookup against a newer release.

    Arguments:
        terms (iterable): ``(accession, label)`` pairs as declared in the
            source file. The label is the original, possibly outdated, name.

    Keyword Arguments:
        obo_translator (OboTranslator): translator pinned to the
            file-declared obo version
        migration (TermMigration): versioned migration mapping (or anything
            accepted by :py:meth:`TermMigration.load`)

    Returns:
        list of dicts: one record per accession, sorted by accession for
            reproducibility. Each record contains the original ``accession``,
            the original label (``name``), the ``unified_accession`` and
            ``unified_name`` after migration, the resolution ``status``
            ("renamed", "deprecated", "current" or "unknown"), the pinned
            ``obo_version`` and the ``mapping_version``.
    """
    migration = TermMigration.load(migration)
    if obo_translator is not None:
        # ensure the pinned obo is parsed; raises FileNotFoundError in
        # offline mode if the pinned version is not available locally
        obo_translator.parseOBO()
    obo_version = getattr(obo_translator, "version", None)
    mapping_version = migration.version if migration is not None else None
    view = {}
    for accession, label in terms:
        if accession in view:
            # first occurrence wins; document order is stable, so the view
            # is reproducible across re-imports
            continue
        record = {
            "accession": accession,
            "name": label,
            "obo_version": obo_version,
            "mapping_version": mapping_version,
        }
        entry = migration.lookup(accession) if migration is not None else None
        if entry is not None:
            record["unified_accession"] = entry["accession"]
            record["unified_name"] = entry["name"]
            record["status"] = entry["status"]
        elif obo_translator is not None and accession in obo_translator.id:
            record["unified_accession"] = accession
            record["unified_name"] = obo_translator.id[accession].get(
                "name", label
            )
            record["status"] = "current"
        else:
            # unknown term: keep the original label, never resolve online
            record["unified_accession"] = accession
            record["unified_name"] = label
            record["status"] = "unknown"
        view[accession] = record
    return [view[key] for key in sorted(view)]
