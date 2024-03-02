# Constants classes
import numpy as np


class MathConstants:
    PI = np.pi


class PhysicalConstants:
    hbar_evs = 6.582119569e-16


class ConversionConstants:
    evperau = 27.211386245988
    angperau = 0.529177210903
    aupergmol = 1822.992667
    cmiperau = 2.1947463136320e5
    evper_amuangpersqsec = 1.0364e-28


# ConstantsManagement class
class ConstantsManagement:
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
