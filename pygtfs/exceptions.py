class PygtfsException(Exception):
    """ A Base class for all pygtfs exceptions """
    pass

class PygtfsValidationError(PygtfsException, ValueError):
    """ Validation error, e.g. int not in range """
    pass

class PygtfsConversionError(PygtfsException):
    """ Failed conversion, e.g. float(value) failed """
    pass


class FrequencyError(PygtfsException):
    """ Base class for frequency expansion errors """
    pass


class InvalidFrequencyError(FrequencyError, ValueError):
    """ A frequency row has invalid times or headway """
    pass


class NonExactFrequencyError(FrequencyError):
    """ Exact departures were requested for a non-exact frequency row """
    pass


class FrequencyConflictError(FrequencyError):
    """ Two frequency periods for the same trip overlap """

    def __init__(self, conflicts):
        self.conflicts = list(conflicts)
        super(FrequencyConflictError, self).__init__(
            '{0} frequency conflict(s): {1}'.format(
                len(self.conflicts),
                '; '.join(str(conflict) for conflict in self.conflicts)))
