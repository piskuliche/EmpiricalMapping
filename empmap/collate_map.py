from empmap.potential import Potential1D
from empmap.sincdvr import DVR
from empmap.constants import ConstantsManagement
import numpy as np


class Collect:
    def __init__(self, file_list=None, file_start=0, file_end=200, file_prefix="Scan_", calc_dir="run_qm/"):
        self.constants = ConstantsManagement()
        if file_list is None:
            self.file_list = np.arange(file_start, file_end)
        else:
            self.file_list = file_list
        self.file_prefix = file_prefix
        self.calc_dir = calc_dir
        return

    def obtain_dvr(self, emax=3.0, xmax=1.3, mass1=2.014, mass2=15.999):
        self.all_dvrs = []
        for file in self.file_list:
            full_prefix = self.calc_dir + "%d/" % file + self.prefix
            pot1d = Potential1D(full_prefix + "rOHs.dat", full_prefix + "energies.dat",
                                full_prefix + "dipoles.dat", full_prefix + "eOHs.dat")
            pot1d.fit_potential_to_poly(3)
            pot1d.fit_dipole_to_poly(2)
            dvr = DVR(pot1d, emax=emax, xmax=xmax, mass1=mass1, mass2=mass2)
            dvr.do_calculation()
            self.all_dvrs.append(dvr)
        return
