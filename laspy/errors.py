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


class StreamAssemblyError(LaspyException):
    pass


class StreamShortReadError(StreamAssemblyError):
    pass


class StreamDeclarationConflict(StreamAssemblyError):
    pass


class StreamOverlapConflict(StreamDeclarationConflict):
    pass


class StreamGapError(StreamAssemblyError):
    pass


class StreamTruncatedError(StreamDeclarationConflict):
    pass


class StreamIncompletePointError(StreamTruncatedError):
    pass
