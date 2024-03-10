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
        -----------
        logname : str
            The name of the log file to read.

        Returns:
        --------
        None

        """
        self.logname = logname
        self.energies = None
        self.dipole = None
        self.polarizability = None
        self.num_points = None
        return

    def description(self):
        """ Prints a description of the GaussianLogFile class

        Notes:
        ------
        This function prints a description of the GaussianLogFile class.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        """
        print("This class is used to read Gaussian log files and extract data from them.")
        print(
            "It is used to extract the energies and the dipole moments from the log files.")
        print("Currently has the following attributes:")
        print("logname: The name of the log file to read.")
        print("energies: The energies from the log file.")
        print("dipole: The dipole moments from the log file.")
        print("polarizability: The polarizabilities from the log file.")
        print("num_points: The number of points in the log file.")
        print("There are %d number of points in the log file." % self.num_points)
        return

    def grab_for_empirical_mapping(self):
        """ Grabs everything from the log file needed for empirical mapping.

        Notes:
        ------

        This function is used to grab everything from the log file that is needed for the DVR. This includes the energies, dipole moments,
        and the polarizabilities.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Raises:
        -------
        ValueError: 
            If no data is found in the log file.

        """
        self.energies = self.grab_energies()
        self.dipole = self.grab_tensor_from_gaussian(
            comparison_choice='dipole')
        self.polarizability = self.grab_tensor_from_gaussian(
            comparison_choice='polarizability')
        self.num_points = len(self.energies)
        return

    def grab_energies(self, expected=None):
        """ Grabs the energies from the log file

        Parameters:
        -----------
        None

        Returns:
        --------
        energies : array_like
            The energies from the log file.

        Raises:
        -------
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
        -----------
        None

        Returns:
        --------
        dipole : array_like
            The dipole moments from the log file.

        Raises:
        -------
        ValueError: 
            If no dipole moments are found in the log file.

        """
        raise DeprecationWarning(
            "This function is deprecated. Use grab_tensor_from_gaussian instead.")
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

    def grab_tensor_from_gaussian(self, comparison_choice='dipole', data_column=2):
        """ Grabs a tensor from the Gaussian log file.

        Notes:
        ------

        This function is used to grab a tensor from the Gaussian log file. It is used to grab the electric dipole moment and the
        dipole polarizability from the log file (but could also do it for the hyperpolarizability, if so desired).

        To get the dipole, set comparison_choice to "dipole". To get the polarizability, set comparison_choice to "polarizability".
        To get the hyperpolarizability, set comparison_choice to "hyperpolarizability".

        Dipole has components x, y, and z, so components should be ["x", "y", "z"].

        Units are:
            [1] au
            [2] debye
            [3] 10**-30 C m (SI)

        Polarizability has components xx, yy, zz, yx, zx, zy, so components should be ["xx", "yy", "zz", "yx", "zx", "zy"].

        Units are:
            [1] au
            [2] 10**-24 cm**3 (esu)
            [3] 10**-30 C**2 m**2 J**-1 (SI)

        Hyperpolarizability has components || (z), _|_(z), x, y, z, ||, xxx, xxy, yxy, yyy, xxz, yxz, yyz, zxz, zyz, zzz
        so components should be ["|| (z)", "_|_(z)", "x", "y", "z", "||", "xxx", "xxy", "yxy", "yyy", "xxz", "yxz", "yyz", "zxz", "zyz", "zzz"]. 

        Units are:
            [1] au
            [2] 10**-30 statvolt**-1 cm**4 (esu)
            [3] 10**-50 C**3 m**3 J**-2 (SI)

        Parameters:
        -----------
        comparison_choice : str
            The type of tensor to grab from the log file.
        data_column : int
            The column to grab the data from.

        Returns:
        --------
        tensor : array_like
            The tensor from the log file.

        Raises:
        -------
        ValueError: 
            If the comparison choice is not recognized.

        """
        if comparison_choice == "dipole":
            comparison_line = "Electric dipole moment (input orientation)"
            components = ["x", "y", "z"]
        elif comparison_choice == "polarizability":
            comparison_line = "Dipole polarizability, Alpha (input orientation)"
            components = ["xx", "yy", "zz", "yx", "zx", "zy"]
        elif comparison_choice == "hyperpolarizability":
            comparison_line = "Hyperpolarizability"
            components = ["|| (z)", "_|_(z)", "x", "y", "z", "||", "xxx",
                          "xxy", "yxy", "yyy", "xxz", "yxz", "yyz", "zxz", "zyz", "zzz"]
        else:
            raise ValueError("Comparison choice not recognized")

        flag_tensor = False
        new_tensor = []
        tensor = []
        with open(self.logname, 'r') as f:
            for line in f:
                # General way of pulling data from the Gaussian log file
                if any(comp in line for comp in components) and flag_tensor:
                    print(line)
                    print(any(comp in line for comp in components))
                    if "|| (z)" in line:
                        data = line.strip().split()[data_column+1]
                    else:
                        data = line.strip().split()[data_column]
                    comp = float(data)
                    new_tensor.append(comp)
                    if len(new_tensor) == len(components):
                        flag_tensor = False
                        tensor.append(new_tensor)
                        new_tensor = []

                # Set the Flag if line is found.
                if comparison_line in line:
                    flag_tensor = True

        return np.array(tensor)


if __name__ == "__main__":
    import sys

    logname = sys.argv[1]
    logfile = GaussianLogFile(logname)
    print(logfile.energies)
    print(logfile.dipole)
    print(logfile.energies.shape)
    print(logfile.dipole.shape)
