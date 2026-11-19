class FeatureNotFoundError(Exception):
    """
    Error to be raised when an ID is not in the database.
    """

    def __init__(self, feature_id):
        Exception.__init__(self)
        self.feature_id = feature_id

    def __str__(self):
        return self.feature_id


class DuplicateIDError(Exception):
    pass


class AttributeStringError(Exception):
    pass


class EmptyInputError(Exception):
    pass


class ParentResolutionError(Exception):
    """
    Raised when a child feature's parent edges cannot be unambiguously
    resolved (missing parent features or conflicting parentage) and the
    caller asked for the unresolved edges to be treated as an error.

    The offending :class:`gffutils.lineage.FeatureLineage` objects are
    available on the ``lineages`` attribute so that callers can inspect the
    raw parent edges instead of having a parent picked for them.
    """

    def __init__(self, lineages, reason="unresolved parentage"):
        self.lineages = list(lineages)
        self.reason = reason
        ids = ", ".join(sorted(lineage.feature_id for lineage in self.lineages))
        message = "%s for feature(s): %s" % (reason, ids)
        super(ParentResolutionError, self).__init__(message)
