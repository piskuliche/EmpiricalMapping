""" This is a class for physical constants

Notes
-----

This class is a simple way to store physical constants in a single place. It is
intended to be used as a parent class for other classes, so that the constants
can be accessed as attributes of the child classes.

Examples
--------

>>> from empmap.constants import ConstantsManagement
>>> constants = ConstantsManagement()
>>> constants.PI

"""

__all__ = ["ConstantsManagement"]

import numpy as np


class MathConstants:
    """ A class for mathematical constants

    Notes
    -----
    This class is a simple way to store mathematical constants in a single place.
    It is intended to be used as a parent class for other classes, so that the
    constants can be accessed as attributes of the child classes.

    """
    PI = np.pi


class PhysicalConstants:
    """ A class for physical constants

    Notes
    -----
    This class is a simple way to store physical constants in a single place.
    It is intended to be used as a parent class for other classes, so that the
    constants can be accessed as attributes of the child classes.

    """

    hbar_evs = 6.582119569e-16  # eV s


class ConversionConstants:
    """ A class for conversion constants

    Notes
    -----
    This class is a simple way to store conversion constants in a single place.
    It is intended to be used as a parent class for other classes, so that the
    constants can be accessed as attributes of the child classes.

    """

    evperau = 27.211386245988  # eV/au
    angperau = 0.529177210903  # angstrom/au
    aupergmol = 1822.992667  # amu/gram/mol
    cmiperau = 2.1947463136320e5  # cm^-1/au
    evper_amuangpersqsec = 1.0364e-28  # eV/(amu*ang/sec^2)


# ConstantsManagement class
class ConstantsManagement:
    """ A class for managing constants

    Notes
    -----
    This class is a wrapper class for the other constants classes. It is intended
    to be used as a parent class for other classes, so that the constants can be
    accessed as attributes of the child classes.

    """

    def __init__(self):
        # Set constants from separate classes as attributes
        for cls in [MathConstants, PhysicalConstants, ConversionConstants]:
            for key, value in cls.__dict__.items():
                if not key.startswith("__"):
                    self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")


if __name__ == "__main__":
    raise SystemExit("This is a module, and should not be run as a script")
