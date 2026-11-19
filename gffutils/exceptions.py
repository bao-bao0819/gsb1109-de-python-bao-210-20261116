class FeatureNotFoundError(Exception):
    """
    Error to be raised when an ID is not in the database.
    """

    def __init__(self, feature_id):
        Exception.__init__(self)
        self.feature_id = feature_id

    def __str__(self):
        return self.feature_id


class DuplicateIDError(ValueError):
    """
    Error to be raised when a duplicate ID is encountered and cannot be
    reconciled (e.g. conflicting core fields for lines sharing an ID in
    GFF3, or any duplicate ID when using merge_strategy="error").

    Subclasses ValueError for backward compatibility.
    """

    pass


class AttributeStringError(Exception):
    pass


class EmptyInputError(Exception):
    pass
