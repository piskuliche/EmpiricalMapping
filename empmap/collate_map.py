from empmap.potential import Potential1D
from empmap.sincdvr import DVR
from empmap.constants import ConstantsManagement
import numpy as np


class EmpiricalMap:
    """ The EmpiricalMap Class that builds an empirical map from a series of QM calculations on a 1D potential.

    This code works on the gaussian calculations setup by emp_setup.py and then fits the potential energy surfaces
    and then goes through and uses those potential energy surfaces for a sinc function discrete variable representation
    (See Colbert, Miller Paper) to obtain the eigenvalues and eigenvectors. For each directory, a value of the key parameters
    w01, w12... etc. are obtained. These are then used to fit an empirical map based on the total data.

    To Do:
        Check that the sinc dvr is being called in the right way.
        Implement the actual empirical map fitting function.

    """

    def __init__(self, file_list=None, file_start=0, file_end=200, file_prefix="scan_", calc_dir="run_qm/"):
        """ Initializes the EmpiricalMap Class

        Args:
            file_list (np.array or None): If not None, gives a list of frames to grab data from.
            file_start (int): Integer value to start grabbing files from [default=0]
            file_end (int): Integer value to stop grabbing files from [default=200]
            file_prefix (str): String that precedes filenames. [default=scan_]
            calc_dir (str): String for the directory name. Should end with /[default=run_qm/]

        Returns:
            None

        """
        self.constants = ConstantsManagement()
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
        self.psi = []
        self.mupsi = []
        self.xpsi = []
        self.Eproj = []
        return

    def create_map_by_fitting(self):
        raise NotImplemented("This function hasn't been implemented yet.")

    def build_base_data(self, **kwargs):
        """ Build data using the DVR approach. 

        Args:
            **kwargs: Keyword arguemnts to be passed to the _obtain_dvrs function.

        Returns:
            None
        """
        dvrs, dvr_read_successful = self._obtain_dvrs(**kwargs)
        self._build_from_dvr(dvrs)
        self.Eproj = self._obtain_eproj()[dvr_read_successful]
        return

    def _build_from_dvr(self, dvrs):
        """ This function builds the data arrays using the DVR approach """
        for dvr in dvrs:
            self.w01.append(dvr.w01)
            self.w12.append(dvr.w12)
            self.psi.append(dvr.psi)
            self.mupsi.append(dvr.mupsi)
            self.xpsi.append(dvr.xpsi)
            self.x01.append(dvr.x01)
            self.x12.append(dvr.x12)

    def _obtain_dvrs(self, emax=3.0, xmax=1.3, mass1=2.014, mass2=15.999, pot_poly_order=5, dip_poly_order=3):
        """ Code to contstruct and obtain eigenvalues and eigenvectors using the DVR approach.

        Args:
            emax (float): Energy maximum in eV
            xmax (float): Position maximum in Angstroms
            mass1 (float): Mass of atom 1 (g/mol)
            mass2 (float): Mass of atom 2 (g/mol)
            pot_poly_order (int): Order of Potential1D potential polynomial
            dip_poly_order (int): Order of Potential1D dipole polynomial

        Returns
            all_dvrs (list): All Calculated DVRs
            dvr_read_successful (np.ndarray): Boolean list of successful and failed calculations.

        """
        all_dvrs = []
        dvr_read_successful = []
        for file in self.file_list:
            try:
                full_prefix = self.calc_dir + "%d/" % file + self.file_prefix
                # Construct the potential object.
                pot1d = Potential1D(full_prefix + "rOHs.dat", full_prefix + "energies.dat",
                                    full_prefix + "dipoles.dat", full_prefix + "eOHs.dat")
                pot1d.fit_potential_to_poly(pot_poly_order)
                pot1d.fit_dipole_to_poly(dip_poly_order)
                # Construct the DVR
                dvr = DVR(pot1d, emax=emax, xmax=xmax,
                          mass1=mass1, mass2=mass2)
                # Do the DVR Calcualtion
                dvr.do_calculation()
                # Store the Data
                all_dvrs.append(dvr)
                dvr_read_successful.append(True)
            except:
                print("Failed to load DVR for file %d" % file)
                raise
                dvr_read_successful.append(False)

        dvr_read_successful = np.array(dvr_read_successful)
        return all_dvrs, dvr_read_successful

    def _obtain_eproj(self):
        """ Reads the projected electric fields from the file based on the file_prefix"""
        eproj = []
        for file in self.file_list:
            full_prefix = self.calc_dir + "%d/" % file + self.file_prefix
            eproj.append(np.loadtxt(full_prefix + "proj_field.dat"))
        return np.array(eproj)
