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


class DimensionNotLoadedError(LaspyException, ValueError):
    """Raised when an operation needs a dimension that was not read.

    This can happen when a compressed LAS 1.4 file (point format >= 6)
    is read with a restrictive :class:`laspy.DecompressionSelection`.
    The bytes of the skipped dimensions contain placeholder values and
    must not be mistaken for real data.
    """

    def __init__(self, missing_dimensions, message: str | None = None) -> None:
        if isinstance(missing_dimensions, str):
            missing = [missing_dimensions]
        else:
            missing = list(missing_dimensions)
        self.missing_dimensions = tuple(missing)
        if message is None:
            if len(missing) == 1:
                message = (
                    f"Dimension '{missing[0]}' was not decompressed/loaded and "
                    "its values are only placeholders"
                )
            else:
                names = ", ".join(repr(name) for name in missing)
                message = (
                    f"Dimensions {names} were not decompressed/loaded and "
                    "their values are only placeholders"
                )
        super().__init__(message)
