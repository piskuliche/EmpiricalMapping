from empmap.constants import ConstantsManagement
import numpy as np
from scipy.linalg import eigh


class DVR:
    """ Class for the Discrete Variable Representation (DVR) Method

    This class is used to calculate the vibrational frequencies and dipole matrix elements
    for a 1D potential using the DVR method. The potential is fit to a polynomial and the
    dipole is fit to a polynomial. The vibrational frequencies and dipole matrix elements
    are calculated using the eigenvalues and eigenvectors of the Hamiltonian matrix.



    Attributes:
        constants (ConstantsManagement): The constants management class
        reduced_mass (float): The reduced mass (g/mol)
        emax (float): The energy cutoff (au)
        xmax (float): The maximum position (au)
        pot1d (Potential1D): The 1D potential object
        ke_pref (float): The kinetic energy prefactor
        hamiltonian (np.array): The Hamiltonian matrix
        evals (np.array): The eigenvalues
        evecs (np.array): The eigenvectors
        w01 (float): The 0-1 vibrational frequency (cm^-1)
        w12 (float): The 1-2 vibrational frequency (cm^-1)
        mu01 (float): The 0-1 dipole matrix element
        mu12 (float): The 1-2 dipole matrix element
        x01 (float): The 0-1 position matrix element
        x12 (float): The 1-2 position matrix element
        delr (float): The grid spacing (au)
        xraw (np.array): The raw grid
        vraw (np.array): The raw potential
        xgrid (np.array): The refined grid
        vgrid (np.array): The refined potential
        ngrid (int): The number of grid points
        _mask_grid (np.array): The mask for the grid

    """

    def __init__(self, pot1d, emax=0.7, xmax=1.4, mass1=1.0, mass2=1.0, reduced_mass=None):
        """ Initialize the DVR class

        Args:
            pot1d (Potential1D): The 1D potential object
            emax (float): The energy cutoff (eV) [Default: 0.7] Converted to au in the class
            xmax (float): The maximum position (angstroms) [Default: 1.4] Converted to au in the class
            mass1 (float): The mass of the first atom (g/mol) [Default: 1.0] 
            mass2 (float): The mass of the second atom (g/mol) [Default: 1.0]
            reduced_mass (float): The reduced mass (g/mol) [Default: None] If None, calculated in the class. Converted to au in the class

        Returns:
            None
        """
        self.constants = ConstantsManagement()
        if reduced_mass is None:
            self.reduced_mass = self.calculate_reduced_mass(
                mass1, mass2)*self.constants.aupergmol
        else:
            self.reduced_mass = reduced_mass * self.constants.aupergmol
        self.emax = emax/self.constants.evperau
        self.xmax = xmax/self.constants.angperau
        self.pot1d = pot1d
        self._setup_grid()
        self.ke_pref = self._kinetic_prefactor()
        self._construct_hamiltonian()

    def description(self):
        """ Print the description of the class

        Args:
            None

        Returns:
            None
        """
        print("DVR Class for 1D Potential")
        print("Reduced Mass: %10.5f au" % self.reduced_mass)
        print("Energy Cutoff: %10.5f au" % self.emax)
        print("Maximum Position: %10.5f au" % self.xmax)
        print("Solved? %s" % hasattr(self, 'evals'))

        print("w01: ", self.w01)
        print("w12: ", self.w12)

        print("mu01: ", self.mu01)
        print("mu12: ", self.mu12)

        print("x01: ", self.x01)
        print("x12: ", self.x12)
        return

    def do_calculation(self):
        self._solve()
        self._set_frequencies()
        self._calculate_dipole_matrix_elements()

    def _calculate_dipole_matrix_elements(self):
        muraw = self.pot1d.mu_fit['poly'](
            self.xraw*self.constants.angperau, *self.pot1d.mu_fit['popt'])

        mu_grid = muraw[self._mask_grid]

        psi0 = self.evecs[:, 0]
        norm0 = np.dot(psi0, psi0)
        psi0 /= np.sqrt(norm0)
        psi1 = self.evecs[:, 1]
        norm1 = np.dot(psi1, psi1)
        psi1 /= np.sqrt(norm1)
        psi2 = self.evecs[:, 2]
        norm2 = np.dot(psi2, psi2)
        psi2 /= np.sqrt(norm2)

        mupsi1 = np.multiply(mu_grid, psi1)
        xpsi1 = np.multiply(self.xgrid, psi1)

        self.mu01 = np.dot(psi0, mupsi1)
        self.mu12 = np.dot(psi2, mupsi1)

        self.x01 = np.dot(psi0, xpsi1)
        self.x12 = np.dot(psi2, xpsi1)

        return

    def _set_frequencies(self):
        """ Set the vibrational frequencies

        Args:
            None

        Returns:
            None
        """
        if not hasattr(self, 'evals'):
            self._solve()
        if len(self.evals) < 3:
            print("Not enough eigenvalues to calculate frequencies")
            return
        self.w01 = (self.evals[1]-self.evals[0])*self.constants.cmiperau
        self.w12 = (self.evals[2]-self.evals[1])*self.constants.cmiperau
        return

    def _solve(self):
        """ Solve the Hamiltonian

        Args:
            None

        Returns:
            None
        """
        self.evals, self.evecs = eigh(self.hamiltonian)
        return

    # Interal Methods
    def _kinetic_prefactor(self):
        """ Calculate the kinetic energy prefactor 

        Units: 
            reduced mass (au) and delr (au)

        Args:
            None

        Returns:
            float: The kinetic energy prefactor in au
        """
        return 1.0/(2.0*self.reduced_mass*self.delr**2)

    def _setup_grid(self):
        """ Set up the grid for the DVR

        Assumptions:
            The grid is evenly spaced
            Radial Coordinate

        Args:
            None

        Returns:
            None
        """
        self.delr = 2.0 * self.constants.PI / \
            (np.sqrt(2.0 * self.reduced_mass * self.emax))
        nraw = int(np.ceil(self.xmax/self.delr))
        xmax_mod = self.delr*np.ceil(self.xmax/self.delr)
        self.xraw = np.linspace(self.delr, xmax_mod, num=nraw, dtype=float)

        # Calculate the potential on the raw grid
        vraw = self.pot1d.pot_fit['poly'](
            self.xraw*self.constants.angperau, *self.pot1d.pot_fit['popt'])

        # Determine the refined grid and potential based on energy cutoff V<emax
        self._mask_grid = np.logical_and(
            vraw <= self.emax, self.xraw <= (self.xmax))  # Check units

        self.xgrid = self.xraw[self._mask_grid]
        self.vgrid = vraw[self._mask_grid]
        self.vgrid = self.vgrid - np.amin(self.vgrid)
        self.ngrid = len(self.xgrid)
        return

    def _construct_hamiltonian(self):
        """ 
        Construct the hamiltionian matrix for the DVR

        Args:
            None

        Returns:
            None

        """
        h = np.zeros((self.ngrid, self.ngrid))

        for i in range(self.ngrid):
            h[i, i] += self.vgrid[i] + self.ke_pref * \
                (np.pi**2/3.0 - 0.5/float(i+1)**2)
            for j in range(i):
                dtmp = (i+1) - (j+1)
                ctmp = float(i+1 + j+1)
                h[i, j] = (-1.0)**dtmp*2.0*self.ke_pref * \
                    (1.0/float(dtmp)**2 - 1.0/ctmp**2)
                h[j, i] = h[i, j]
        self.hamiltonian = h

    @staticmethod
    def calculate_reduced_mass(m1, m2):
        """ Calculate the reduced mass

        Args:
            m1 (float): The mass of the first atom (g/mol)
            m2 (float): The mass of the second atom (g/mol)

        Returns:
            float: The reduced mass (g/mol)
        """
        return m1*m2/(m1+m2)