from __future__ import annotations


class Error(Exception):
    pass


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class ResolveError(Error):
    pass


class NullPointerDereference(Error):
    pass


class AddressMappingError(Error):
    """Raised when a virtual address cannot be resolved to source bytes."""


class AmbiguousMappingError(AddressMappingError):
    """Raised when overlapping mappings provide different bytes for the same virtual address."""


class ArraySizeError(Error):
    pass


class ExpressionParserError(Error):
    pass


class UnknownIdentifierError(ExpressionParserError):
    pass
