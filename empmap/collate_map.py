"""
This module contains the EmpiricalMap class that is used to build an empirical map from a series of QM calculations on a 1D potential.

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


class EmpiricalMap:
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

        self.w01 = []
        self.w12 = []
        self.x01 = []
        self.x12 = []
        self.mu01 = []
        self.mu12 = []
        self.psi = []
        self.mupsi1 = []
        self.xpsi1 = []
        self.Eproj = []
        self.dmu = []
        self.dmu_num = []
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

    @staticmethod
    def r2_score(xdata, y_actual, poly, popt, scale_x=1.0, scale_y=1.0):
        """ Calculate the R^2 score. 

        Notes:
        ------

        The R^2 score is a measure of how well the data fits the polynomial. It is calculated as:

        R^2 = 1 - (SS_res/SS_tot)

        Where SS_res is the sum of the squares of the residuals and SS_tot is the sum of the squares of the total data.

        It should be noted that for high-order polynomials, the R^2 score can be artificially high. This is because the polynomial
        can be made to fit the data very well, but it may not be a good representation of the data. This is known as overfitting.


        Parameters:
        -----------
        xdata : array_like
            The x data.
        y_actual : array_like
            The actual values of the data.
        poly : function
            The polynomial function.
        popt : array_like
            The polynomial coefficients.

        Returns:
        --------
        r2_score : float
            The R^2 score.

        """
        xdata, y_actual = np.array(xdata)*scale_x, np.array(y_actual)*scale_y
        residuals = y_actual - poly(xdata, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
        return 1 - (ss_res/ss_tot)

    def fit_full_empirical_map(self, order_omega=2, order_x=1, order_mu=1, experimental_w01=3707, sigma_pos=None):
        """ Fits the full empirical map.

        Parameters:
        -----------
        order_omega : int
            The order of the polynomial to fit the w01 and w12 parameters.
        order_x : int
            The order of the polynomial to fit the x01 and x12 parameters.

        Returns:
        --------
        None

        """
        constants = ConstantsManagement()

        w01_scale = self.w01[-1]/experimental_w01
        self.map_fit_parameters = {}
        # Fit the first fundamenetal frequency, w01
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "Eproj", "w01", order_omega, scale_factor2=w01_scale, sigma_pos=sigma_pos)
        r2_score = self.r2_score(self.Eproj, self.w01,
                                 poly, popt, scale_y=w01_scale)
        self.map_fit_parameters['w01'] = (popt, pcov, r2_score)

        # Fit the second fundamenetal frequency, w12
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "Eproj", "w12", order_omega, scale_factor2=w01_scale, sigma_pos=sigma_pos)
        r2_score = self.r2_score(self.Eproj, self.w12,
                                 poly, popt, scale_y=w01_scale)
        self.map_fit_parameters['w12'] = (popt, pcov, r2_score)

        # Fit the first positon matrix element, x01
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "w01", "x01", order_x, scale_factor1=w01_scale, scale_factor2=constants.angperau)
        r2_score = self.r2_score(
            self.w01, self.x01, poly, popt, scale_x=w01_scale, scale_y=constants.angperau)
        self.map_fit_parameters['x01'] = (popt, pcov, r2_score)

        # Fit the second positon matrix element, x12
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "w12", "x12", order_x, scale_factor1=w01_scale, scale_factor2=constants.angperau)
        r2_score = self.r2_score(
            self.w12, self.x12, poly, popt, scale_x=w01_scale, scale_y=constants.angperau)
        self.map_fit_parameters['x12'] = (popt, pcov, r2_score)

        # Fit the first dipole matrix derivative, mu'
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "Eproj", "dmu_num", order_mu, sigma_pos=sigma_pos)
        r2_score = self.r2_score(self.Eproj, self.dmu_num, poly, popt)
        self.map_fit_parameters['dmu_num'] = (popt, pcov, r2_score)

        # Fit the first dipole matrix derivative, mu'
        poly, popt, pcov = self.fit_attribute_vs_attribute(
            "Eproj", "dmu", order_mu, sigma_pos=None)
        r2_score = self.r2_score(self.Eproj, self.dmu, poly, popt)
        self.map_fit_parameters['dmu'] = (popt, pcov, r2_score)

        # Fit the first dipole matrix derivative, mu'
        data1 = self.Eproj
        data2 = np.divide(self.dmu_num, self.dmu_num[-1])
        poly, popt, pcov = self.fit_data_vs_data(
            data1, data2, order_mu, label1='E', label2='dmu_num_scaled')
        r2_score = self.r2_score(
            data1, data2, poly, popt)
        self.map_fit_parameters['dmu_num_scaled'] = (popt, pcov, r2_score)

        # New test
        data1 = self.Eproj
        data2 = np.divide(self.dmu, self.dmu[-1])
        poly, popt, pcov = self.fit_data_vs_data(
            data1, data2, order_mu, label1='E', label2='dmu_scaled')
        r2_score = self.r2_score(
            data1, data2, poly, popt)
        self.map_fit_parameters['dmu_scaled'] = (popt, pcov, r2_score)
        return

    def fit_attribute_of_map(self, attribute, order, scale_factor=1.0, display_plot=False):
        """ Fits an attribute of the map to a polynomial of order order.

        Parameters:
        -----------
        attribute : str
            The attribute to fit. This should be one of the attributes of the EmpiricalMap class.
        order : int
            The order of the polynomial to fit.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError: If the attribute is not present in the class.

        """
        # Grab the correct polynomial to fit.
        poly = mu_fit_selector(order)

        # Check if the attribute is present in the class.
        if getattr(self, attribute) is None:
            raise ValueError(
                f"The attribute {attribute} is not present in the class")

        values_to_fit = np.array(getattr(self, attribute))
        popt, pcov = curve_fit(poly, self.Eproj, values_to_fit*scale_factor)
        self._print_fit(popt, attribute)
        if display_plot:
            scaled_values = values_to_fit*scale_factor
            self._display_fit(poly, popt, scaled_values, attribute)
        return poly, popt, pcov

    def fit_attribute_vs_attribute(self, attribute1, attribute2, order, scale_factor1=1.0, scale_factor2=1.0, sigma_pos=None):
        poly = mu_fit_selector(order)

        if getattr(self, attribute1) is None:
            raise ValueError(
                f"The attribute {attribute1} is not present in the class")
        if getattr(self, attribute2) is None:
            raise ValueError(
                f"The attribute {attribute2} is not present in the class")

        values_to_fit1 = np.array(getattr(self, attribute1))*scale_factor1
        values_to_fit2 = np.array(getattr(self, attribute2))*scale_factor2
        if sigma_pos is not None:
            sigma = np.ones(len(values_to_fit1))
            sigma[sigma_pos] = 0.01
            popt, pcov = curve_fit(poly, values_to_fit1, values_to_fit2,
                                   sigma=sigma)
        else:
            popt, pcov = curve_fit(poly, values_to_fit1, values_to_fit2)
        self._print_fit(popt, attribute2, label=attribute1)
        return poly, popt, pcov

    def fit_data_vs_data(self, data1, data2, order, label1='E', label2='w01', scale_factor1=1.0, scale_factor2=1.0, sigma_pos=None):
        """ """
        poly = mu_fit_selector(order)

        values_to_fit1 = np.array(data1)*scale_factor1
        values_to_fit2 = np.array(data2)*scale_factor2
        if sigma_pos is not None:
            sigma = np.ones(len(values_to_fit1))
            sigma[sigma_pos] = 0.01
            popt, pcov = curve_fit(poly, values_to_fit1, values_to_fit2,
                                   sigma=sigma)
        else:
            popt, pcov = curve_fit(poly, values_to_fit1, values_to_fit2)
        self._print_fit(popt, label2, label=label1)
        return poly, popt, pcov

    def _print_fit(self, popt, attribute, label='E'):
        """ Prints the fit to the screen. """
        if len(popt) == 2:
            print("%s = %10.10f + %10.10f*%s" %
                  (attribute, popt[0], popt[1], label))
        elif len(popt) == 3:
            print("%s = %10.10f + %10.10f*%s + %10.10f*%s^2" % (attribute,
                  popt[0], popt[1], label, popt[2], label))
        return

    def _display_fit(self, poly, popt, values, attribute):
        """ Displays the fit to the screen. """
        fig = plt.figure()
        es = np.linspace(self.Eproj.min(), self.Eproj.max(), 100)
        plt.scatter(self.Eproj, values)
        plt.plot(es, poly(es, *popt))
        plt.xlabel("E")
        plt.ylabel(f"{attribute}")
        plt.show()

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

    def _build_from_dvr(self, dvrs):
        """ This function builds the data arrays using the DVR approach """

        attributes = ['w01', 'w12', 'psi', 'mupsi1',
                      'xpsi1', 'x01', 'x12', 'mu01', 'mu12']
        for dvr in dvrs:
            for attribute in attributes:
                if not hasattr(self, attribute):
                    setattr(self, attribute, [])
                getattr(self, attribute).append(
                    np.abs(getattr(dvr, attribute)))
            self.dmu.append(np.abs(dvr.pot1d.mu_fit['dmu/dr_r0']))
            self.dmu_num.append(np.abs(dvr.pot1d.mu_fit['dmu_num']))
        attributes.extend(['dmu', 'dmu_num'])
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
            except Exception:
                print("Failed to load DVR for file %d" % file)
                dvr_read_successful.append(False)
                fail_count += 1
                if fail_count > max_fail:
                    raise
        if os.path.exists(f"{self.calc_dir}gas/"):
            try:
                dvr = self.obtain_dvr("gas", emax=emax, xmax=xmax, mass1=mass1, mass2=mass2,
                                      pot_poly_order=pot_poly_order, dip_poly_order=dip_poly_order)
                # Store the Data
                all_dvrs.append(dvr)
                dvr_read_successful.append(True)
            except Exception:
                print("Failed to load DVR for file gas")
                dvr_read_successful.append(False)
                fail_count += 1
                if fail_count > max_fail:
                    raise RuntimeError(
                        "Too many failed DVR calculations. Please increase max_fail, or check the calculations.")

        dvr_read_successful = np.array(dvr_read_successful)
        return all_dvrs, dvr_read_successful

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
