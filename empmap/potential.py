""" This module contains the Morse and Potential1D classes

Notes:
------
The Morse class is used to fit the Morse potential to the potential energy data. 
The Potential1D class is used to read the data from the files and then fit the potential energy and dipole moment to a polynomial.

Examples:
---------
>>> from empmap.potential import Morse
>>> morse = Morse()
>>> morse.fit_parameters(r, E, mu, p0=[100, 10, 1.0])

>>> from empmap.potential import Potential1D
>>> pot = Potential1D("rOH_file", "pot_file", "dip_file", "eOH_file")
>>> pot.fit_potential_to_poly(3)
>>> pot.fit_dipole_to_poly(2)

"""
__all__ = ["Morse", "Potential1D"]


import numpy as np
from scipy.optimize import curve_fit

from empmap.constants import ConstantsManagement
from empmap.poly_fit import poly_fit_selector, mu_fit_selector


class Morse:
    def __init__(self):
        """ Initialize the Morse potential class
        """
        self.constants = ConstantsManagement()
        self.de = None
        self.alpha = None
        self.re = None
        pass

    def __repr__(self):
        """ Return the Morse potential representation"""
        return "Morse(de={}, alpha={}, re={})".format(self.de, self.alpha, self.re)

    def description(self):
        """ Print the Morse potential parameters """
        print("Morse potential with \nde={}, \nalpha={}, \nre={}".format(
            self.de, self.alpha, self.re))

    def fit_parameters(self, r, E, mu, p0=[100, 10, 1.0]):
        """ Fit the Morse potential parameters

        Notes:
        ------
        This function is used to fit the Morse potential parameters to the potential energy data.

        Parameters:
        -----------
        r : array_like
            The bond distance values [Distance Unit]
        E : array_like
            The potential energy values [Energy Unit]
        mu : float
            The reduced mass of the oscillator [Mass Unit]
        p0 : array_like
            The initial guess for the Morse potential parameters (de, a, re)

        Returns:
        --------
        popt : array_like
            The optimized parameters
        pcov : array_like
            The covariance matrix

        """
        E = E - E.min()
        popt, pcov = curve_fit(self.energy, r, E, p0=p0)
        self.de = popt[0]
        self.alpha = popt[1]
        self.re = popt[2]
        self.mu = mu

        self.B = self.morse_Bfactor(self.alpha, self.mu, self.de)
        return popt, pcov

    def energy(self, r, de, a, re):
        """ Calculate the Morse potential

        Notes:
        ------
        This function is used to calculate the Morse potential.

        The Morse potential is given by:
        V(r) = de*(1 - exp(-a*(r - re)))^2

        Parameters:
        ----------
            de : float
                The dissociation energy [Energy Unit]
            a : float
                The Morse potential parameter [1/Distance Unit]
            re : float
                The equilibrium bond length [Distance Unit]

        Returns:
        -------
            float: The Morse potential [Energy Unit]

        """
        return de * (1.0 - np.exp(-a * (r - re)))**2

    def morse_Bfactor(self, alpha, mu, de):
        """ Calculate the Morse potential B factor

        Notes:
        ------
        This function is used to calculate the Morse potential B factor.

        The Morse potential B factor is given by:
        B = sqrt(alpha^2*hbar^2/(2*mu*de))

        Parameters:
        ----------
        alpha : float
            The Morse potential parameter [1/Distance Unit]
        mu : float
            The reduced mass of the oscillator [Mass Unit]
        de : float
            The dissociation energy [Energy Unit]

        Returns:
        -------
        float: The Morse potential B factor [Energy Unit]

        """

        hbar = self.constants.hbar_evs
        return np.sqrt(alpha**2.*hbar**2./(2.*mu*de)/self.constants.evper_amuangpersqsec)

    @staticmethod
    def morse_eigenvalues(de, B, n):
        """ Calculate the Morse potential eigenvalues

        Notes:
        ------
        This function is used to calculate the Morse potential eigenvalues.

        The Morse potential eigenvalues are given by:
        E_n = de*(n+0.5)*(2-B*(n+0.5))

        Parameters:
        ----------
        de : float
            The dissociation energy [Energy Unit]
        B : float
            The Morse potential B factor [Energy Unit]
        n : int
            The quantum number

        Returns:
        -------
        float: The Morse potential eigenvalues [Energy Unit]

        """

        return de*B*(n+0.5)*(2-B*(n+0.5))


class Potential1D:
    def __init__(self, rOH_file, pot_file, dip_file, eOH_file):
        """ Initialize the Potential1D class

        Notes:
        ------
        This class is used to read the data from the files and then fit the potential energy and dipole moment to a polynomial.

        Parameters:
        -----------
        rOH_file : str
            The file containing the rOH values [Angstroms]
        pot_file : str
            The file containing the potential energy values [eV]
        dip_file : str
            The file containing the dipole values [D]
        eOH_file : str
            The file containing the eOH values [Unitless]

        Returns:
        --------
        None

        """
        self.rOH = None
        self.pot_energy = None
        self.mux, self.muy, self.muz = None, None, None
        self._read_data(rOH_file, pot_file, dip_file, eOH_file)
        self.ndata = len(self.rOH)
        self._project_mu()
        self.pot_fit = {}
        self.mu_fit = {}

        self.constants = ConstantsManagement()

    def fit_potential_to_poly(self, order):
        """ Fit the potential to a polynomial

        Notes:
        ------
        This function is used to fit the potential energy to a polynomial.


        Parameters:
        -----------
        order : int
            The order of the polynomial

        Returns:
        --------
            None

        """
        poly = poly_fit_selector(order)
        popt, pcov = curve_fit(poly, self.rOH, self.pot_energy)
        self._display_polyfit(order, poly, popt)
        self.pot_fit['order'] = order
        self.pot_fit['popt'] = popt
        self.pot_fit['pcov'] = pcov
        self.pot_fit['poly'] = poly
        self.pot_fit['V0'] = popt[0]
        self.pot_fit['r0'] = popt[1]
        self.pot_fit['k'] = popt[2]
        self.pot_fit['c'] = popt[3:]
        return

    def fit_dipole_to_poly(self, order):
        """ Fit the dipole to a polynomial

        Notes:
        ------
        This function is used to fit the dipole moment to a polynomial.

        Parameters:
        -----------
        order : int
            The order of the polynomial

        Returns:
        --------
        None


        """
        mu_poly = mu_fit_selector(order)
        if order > 1:
            mu_deriv_poly = mu_fit_selector(order-1)
        popt, pcov = curve_fit(mu_poly, self.rOH, self.mu)
        if order > 1:
            dmu_popt = self._deriv_of_polyfit(popt)

        # Find the closest data to the eq. bond distance
        drOH = np.abs(self.rOH - self.pot_fit['r0'])
        r0_index = np.argmin(drOH)
        dmu_num = np.abs((self.mu[r0_index+1]-self.mu[r0_index-1]) /
                         (self.rOH[r0_index+1]-self.rOH[r0_index-1]))

        dmu = mu_deriv_poly(self.pot_fit['r0'], *dmu_popt)
        self._display_mu_polyfit(order, popt, dmu, dmu_num)

        self.mu_fit['mu0'] = popt[0]
        self.mu_fit['poly'] = mu_poly
        self.mu_fit['order'] = order
        self.mu_fit['popt'] = popt
        self.mu_fit['dmu_popt'] = dmu_popt
        self.mu_fit['dmu_num'] = dmu_num
        self.mu_fit['mu_poly'] = mu_poly
        self.mu_fit['mu_deriv_poly'] = mu_deriv_poly
        if order == 1:
            self.mu_fit['dmu/dr_r0'] = np.abs(popt[1])
        else:
            self.mu_fit['dmu/dr_r0'] = np.abs(dmu)
            self.mu_fit['dmu/dr_num'] = popt[2]

        if len(popt) > 2:
            self.mu_fit['c'] = popt[2:]

        return

    def fit_to_morse(self, reduced_mass):
        """ Fit the potential to a Morse potential

        Notes:
        ------
        This function is used to fit the potential energy to a Morse potential.

        Parameters:
        -----------
        reduced_mass : float
            The reduced mass of the oscillator [Mass Unit]

        Returns:
        --------
        None

        """
        morse = Morse()
        morse.fit_parameters(self.rOH, self.pot_energy,
                             reduced_mass)
        morse.description()

    def _project_mu(self):
        """ Project the dipole moment onto the eOH vector

        Notes:
        ------
        This function is used to project the dipole moment onto the eOH vector.

        Returns:
        --------
        None

        """
        self.mu = np.zeros(self.ndata)
        for i in range(self.ndata):
            self.mu[i] += self.mux[i]*self.eOH[0] + \
                self.muy[i]*self.eOH[1] + self.muz[i]*self.eOH[2]
        return

    def _deriv_of_polyfit(self, popt):
        """ Calculate the derivative of the fit

        Notes:
        ------
        This function is used to calculate the derivative of the fit.

        Parameters:
        -----------
        popt : array_like
            The optimized parameters

        Returns:
        --------
        dmufit_popt : array_like
            The derivative of the fit parameters
        """
        dmufit_popt = None
        deriv_order = len(popt)-1
        if deriv_order > 1:
            dmufit_popt = np.zeros(deriv_order)
            for i in range(deriv_order):
                dmufit_popt[i] = popt[i+1]
        return dmufit_popt

    def _display_polyfit(self, order, poly, popt, verbose=False):
        """ Display the polynomial fit

        Notes:
        ------
        This function is used to display the polynomial fit.

        Parameters:
        -----------
        order : int
            The order of the polynomial
        poly : function
            The polynomial function
        popt : array_like
            The optimized parameters
        verbose : bool

        Returns:
        --------
        None

        """

        vf = poly(self.rOH, *popt)
        print("# **************************************************")
        print('#  V0 = %13.8f' % popt[0]+' eV')
        print('#  r0 = %13.8f' % popt[1]+' Angs')
        print('#  k  = %13.8f' % popt[2]+' eV/Angs^2')
        for j in range(3, int(order)+1):
            print('#  c'+str(j)+' = %13.8f eV/Angs^' % popt[j]+str(j))
        if verbose == True:
            print('# rOH (Ang) v_actual    v_fit (eV)')
            for k in range(0, len(self.rOH)):
                print(
                    f"{self.rOH[k]:.8f}  {self.pot_energy[k]:.8f}  {vf[k]:.8f}")
        print("# **************************************************")

    def _display_mu_polyfit(self, order, popt, dmu, dmu_num, verbose=False):
        """ Display the dipole moment polynomial fit

        Notes:
        ------
        This function is used to display the dipole moment polynomial fit.

        Parameters:
        -----------
        order : int
            The order of the polynomial
        popt : array_like
            The optimized parameters
        dmu : float
            The derivative of the fit
        dmu_num : float
            The numerical derivative of the fit
        verbose : bool
            Flag to print the verbose output

        Returns:
        --------
        None

        """
        print("# **************************************************")
        if (order == 1):
            print('#  mu0       = %13.8f' % popt[0]+' D')
            print('#  dmu/dr_r0 = %13.8f' % (np.abs(popt[1]))+' D/Angs')
        else:
            print('# mu0        = %13.8f' % popt[0]+' D')
            print('# dmu/dr_r0  = %13.8f' % (np.abs(dmu))+' D/Angs')
            print('# dmu/dr_num = %13.8f' % dmu_num+' D/Angs')
            for j in range(1, order+1):
                print('#  c'+str(j)+' = %13.8f D/Angs^' %
                      popt[j]+str(j))
        print("# **************************************************")

    def _read_data(self, rOH_file, pot_file, dip_file, eOH_file):
        """ Read the data from the files

        Notes:
        ------
        This function is used to read the data from the files.

        These files contain the rOH, potential energy, dipole, and eOH values.

        Parameters:
        -----------
        rOH_file : str
            The file containing the rOH values [Angstroms]
        pot_file : str
            The file containing the potential energy values [eV]
        dip_file : str
            The file containing the dipole values [D]
        eOH_file : str
            The file containing the eOH values [Unitless]

        Returns:
        --------
            None

        Raises:
        -------
        ValueError:
            If there is an error reading the files

        """
        try:
            self.rOH = np.genfromtxt(
                rOH_file, dtype=float, usecols=(0), unpack=True)
        except:
            raise ValueError("Error reading rOH file %s" % rOH_file)
        try:
            self.pot_energy = np.genfromtxt(
                pot_file, dtype=float, usecols=(0), unpack=True)
            self.pot_energy = self.pot_energy - self.pot_energy.min()
        except:
            raise ValueError("Error reading potential file %s" % pot_file)
        try:
            self.mux, self.muy, self.muz = np.genfromtxt(
                dip_file, dtype=float, usecols=(0, 1, 2), unpack=True)
        except:
            raise ValueError("Error reading dipole file %s" % dip_file)
        try:
            self.eOH = np.genfromtxt(
                eOH_file, dtype=float, usecols=(0), unpack=True)
        except:
            raise ValueError("Error reading eOH file %s " % eOH_file)
        return


if __name__ == "__main__":
    print("Setting up the potential class")

    pot = Potential1D("../newmap/1/scan_rOHs.dat", "../newmap/1/scan_energies.dat",
                      "../newmap/1/scan_dipoles.dat", "../newmap/1/scan_eOHs.dat")
    import numpy as np
    import matplotlib.pyplot as plt
    pot.fit_to_morse(34/19)
    pot.fit_potential_to_poly(3)
    pot.fit_dipole_to_poly(2)
