"""All the custom exceptions types"""


class LaspyException(Exception):
    pass


class UnknownExtraType(LaspyException):
    pass


class PointFormatNotSupported(LaspyException):
    pass


class FileVersionNotSupported(LaspyException):
    pass


class LazError(LaspyException):
    pass


class IncompatibleDataFormat(LaspyException):
    pass


class CrsError(LaspyException):
    """Base class for coordinate reference system related errors."""


class CrsUnresolvedError(CrsError):
    """Raised when CRS metadata exists but no usable CRS could be selected."""


class CrsMissingError(CrsUnresolvedError):
    """Raised when no CRS metadata could be found."""


class CrsConflictError(CrsUnresolvedError):
    """Raised when CRS metadata sources disagree and the ambiguity is unresolved."""


class CrsTransformError(CrsError):
    """Raised when coordinates cannot be transformed between CRS."""
