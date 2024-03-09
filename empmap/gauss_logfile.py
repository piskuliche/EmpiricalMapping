"""
This module contains a class to read Gaussian log files and extract data from them.

Notes:
------
This class is used to read Gaussian log files and extract data from them. It is used to extract the energies and the dipole moments from the log files.

Examples:
---------
>>> from empmap.gauss_logfile import GaussianLogFile
>>> logfile = GaussianLogFile("file.log")
>>> print(logfile.energies)
>>> print(logfile.dipole)
"""
__all__ = ["GaussianLogFile"]


import numpy as np


class GaussianLogFile:
    def __init__(self, logname):
        """ Initializes the GaussianLogFile class

        Notes:
        ------
        This class is used to read Gaussian log files and extract data from them. It is used to extract the energies and the dipole moments from the log files.

        Parameters:
        ----------
        logname : str
            The name of the log file to read.

        Returns:
        -------
        None

        """
        self.logname = logname
        self.energies = self.grab_energies()
        self.dipole = self.grab_dipole()

    def grab_energies(self, expected=None):
        """ Grabs the energies from the log file

        Parameters:
        ----------
        None

        Returns:
        -------
        energies : array_like
            The energies from the log file.

        Raises:
        ------
        ValueError: 
            If no energies are found in the log file.

        """
        energies = []
        with open(self.logname, 'r') as f:
            for line in f:
                if "SCF Done" in line:
                    energy = float(line.split()[4])
                    energies.append(energy)

        if len(energies) == 0:
            raise ValueError("No energies found in the log file")

        if expected is not None:
            if len(energies) != expected:
                raise ValueError("Expected %d energies, but found %d" %
                                 (expected, len(energies)))

        return np.array(energies)

    def grab_dipole(self):
        """ Grabs the dipole moments from the log file

        Parameters:
        ----------
        None

        Returns:
        -------
        dipole : array_like
            The dipole moments from the log file.

        Raises:
        ------
        ValueError: 
            If no dipole moments are found in the log file.

        """

        dipole = []
        prev_line = ""
        with open(self.logname, 'r') as f:
            for line in f:
                if "X= " in line and "Dipole moment (field" in prev_line:
                    x = float(line.split()[1])
                    y = float(line.split()[3])
                    z = float(line.split()[5])
                    dipole.append([x, y, z])
                prev_line = line

        if len(dipole) == 0:
            raise ValueError("No dipole found in the log file")

        return np.array(dipole)


if __name__ == "__main__":
    import sys

    logname = sys.argv[1]
    logfile = GaussianLogFile(logname)
    print(logfile.energies)
    print(logfile.dipole)
    print(logfile.energies.shape)
    print(logfile.dipole.shape)
