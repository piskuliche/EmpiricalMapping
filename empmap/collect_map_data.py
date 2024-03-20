"""
This module contains the CollectMapData class that is used to build an empirical map from a series of QM calculations on a 1D potential.

Notes:
------
This code works on the gaussian calculations setup by emp_setup.py and then fits the potential energy surfaces. 
It then goes through and uses those potential energy surfaces for a sinc function discrete variable representation (See Colbert, Miller Paper) 
to obtain the eigenvalues and eigenvectors. For each directory, a value of the key parameters w01, w12... etc. are obtained. 
These are then used to fit an empirical map based on the total data.

To Do:
------
    1) Write the map files consistent with frequencymap.org

"""

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


from empmap.potential import Potential1D
from empmap.sincdvr import DVR
from empmap.constants import ConstantsManagement
from empmap.poly_fit import mu_fit_selector


class CollectMapData:
    """ The EmpiricalMap Class that builds an empirical map from a series of QM calculations on a 1D potential.

    Notes:
    ------
    This code works on the gaussian calculations setup by emp_setup.py and then fits the potential energy surfaces
    and then goes through and uses those potential energy surfaces for a sinc function discrete variable representation
    (See Colbert, Miller Paper) to obtain the eigenvalues and eigenvectors. For each directory, a value of the key parameters
    w01, w12... etc. are obtained. These are then used to fit an empirical map based on the total data.

    """

    def __init__(self, file_list=None, file_start=0, file_end=200, file_prefix="scan_", calc_dir="run_qm/"):
        """ Initializes the EmpiricalMap Class

        Notes:
        ------
        The EmpiricalMap class is used to build an empirical map from a series of QM calculations on a 1D potential.
        It pulls data from each of the directories used by the QM calculations and then uses that data to build the empirical map.
        This generally involves fitting the potential energy surfaces and then using those potential energy surfaces for a sinc function
        discrete variable representation (See Colbert, Miller Paper) to obtain the eigenvalues and eigenvectors. For each directory,
        a value of the key parameters w01, w12... etc. are obtained. These are then used to fit an empirical map based on the total data.


        Parameters:
        -----------
        file_list : array_like
            List of file numbers to use.
        file_start : int
            Starting file number to use. [Default: 0]
        file_end : int
            Ending file number to use. [Default: 200]
        file_prefix : str
            Prefix of the file name. [Default: "scan_"]
        calc_dir : str
            Directory where the calculations are located. [Default: "run_qm/"]

        Returns:
            None

        """
        self.constants = ConstantsManagement()

        # Set the file list, either by the input or by the range.
        if file_list is None:
            self.file_list = np.arange(file_start, file_end)
        else:
            self.file_list = file_list

        self.file_prefix = file_prefix
        self.calc_dir = calc_dir

        return

    def save_self(self, filename):
        """ Save the EmpiricalMap to a file. """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        return

    @classmethod
    def load_self(cls, filename):
        """ Load the EmpiricalMap from a file. """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def build_base_data(self, **kwargs):
        """ Build data using the DVR approach. 

        Paramters:
        ----------
        kwargs : arguments
            Keyword arguemnts to be passed to the _obtain_dvrs function.

        Returns:
        --------
        None
        """
        dvrs, dvr_read_successful = self._obtain_dvrs(**kwargs)
        self._build_from_dvr(dvrs)
        self.Eproj = self._obtain_eproj()[dvr_read_successful]
        return

    def obtain_dvr(self, file,  emax=3.0, xmax=1.3, mass1=2.014, mass2=15.999, pot_poly_order=5, dip_poly_order=3):
        """ Obtain a DVR for a given file. """
        full_prefix = f"{self.calc_dir}{file}/{self.file_prefix}"
        # Construct the potential object.
        pot1d = Potential1D(
            f"{full_prefix}rOHs.dat",
            f"{full_prefix}energies.dat",
            f"{full_prefix}dipoles.dat",
            f"{full_prefix}eOHs.dat",
        )
        pot1d.fit_potential_to_poly(pot_poly_order)
        pot1d.fit_dipole_to_poly(dip_poly_order)
        # Construct the DVR
        dvr = DVR(pot1d, emax=emax, xmax=xmax, mass1=mass1, mass2=mass2)
        # Do the DVR Calcualtion
        dvr.do_calculation()
        return dvr

    def _build_from_dvr(self, dvrs):
        """ This function builds the data arrays using the DVR approach """

        attributes = ['w01', 'w12', 'psi', 'mupsi1',
                      'xpsi1', 'x01', 'x12', 'mu01', 'mu12', 'dmu']
        for dvr in dvrs:
            # Loop over normal attributes
            for attribute in attributes:
                if not hasattr(self, attribute):
                    setattr(self, attribute, [])
                getattr(self, attribute).append(
                    np.abs(getattr(dvr, attribute)))
        for attribute in attributes:
            try:
                setattr(self, attribute, np.array(getattr(self, attribute)))
            except ValueError:
                print(f"Failed to set attribute {attribute} as an array")

    def _obtain_dvrs(self, emax=3.0, xmax=1.3, mass1=2.014, mass2=15.999, pot_poly_order=5, dip_poly_order=3, max_fail=10):
        """ Code to contstruct and obtain eigenvalues and eigenvectors using the DVR approach.

        Parameters:
        -----------
            emax : float
                Energy maximum in eV
            xmax : float
                Position maximum in Angstroms
            mass1 : float
                Mass of atom 1 (g/mol)
            mass2 : float
                Mass of atom 2 (g/mol)
            pot_poly_order : int
                Order of Potential1D potential polynomial
            dip_poly_order : int
                Order of Potential1D dipole polynomial

        Returns
        -------
            all_dvrs : list 
                All Calculated DVRs
            dvr_read_successful : np.ndarray
                Boolean list of successful and failed calculations.

        """
        all_dvrs = []
        dvr_read_successful = []
        fail_count = 0
        for file in self.file_list:
            try:
                dvr = self.obtain_dvr(file, emax=emax, xmax=xmax, mass1=mass1, mass2=mass2,
                                      pot_poly_order=pot_poly_order, dip_poly_order=dip_poly_order)
                # Store the Data
                all_dvrs.append(dvr)
                dvr_read_successful.append(True)
            except Exception as e:
                print("Failed to load DVR for file %d" % file)
                dvr_read_successful.append(False)
                fail_count += 1
                if fail_count > max_fail:
                    raise RuntimeError(
                        "Too many failed DVR calculations. Please increase max_fail, or check the calculations."
                    ) from e
        # Now do the gas phase calculation
        if os.path.exists(f"{self.calc_dir}gas/"):
            try:
                dvr = self.obtain_dvr("gas", emax=emax, xmax=xmax, mass1=mass1, mass2=mass2,
                                      pot_poly_order=pot_poly_order, dip_poly_order=dip_poly_order)
                # Store the Data
                all_dvrs.append(dvr)
                dvr_read_successful.append(True)
            except Exception as e:
                print("Failed to load DVR for file gas")
                dvr_read_successful.append(False)
                fail_count += 1
                if fail_count > max_fail:
                    raise RuntimeError(
                        "Too many failed DVR calculations. Please increase max_fail, or check the calculations."
                    ) from e

        dvr_read_successful = np.array(dvr_read_successful)
        return all_dvrs, dvr_read_successful

    def _obtain_eproj(self):
        """ Reads the projected electric fields from the file based on the file_prefix """
        eproj = []
        for file in self.file_list:
            full_prefix = self.calc_dir + "%d/" % file + self.file_prefix
            eproj.append(np.loadtxt(f"{full_prefix}proj_field.dat"))
        if os.path.exists(f"{self.calc_dir}gas/"):
            eproj.append(np.loadtxt(
                f"{self.calc_dir}gas/{self.file_prefix}proj_field.dat"))
        return np.array(eproj)
