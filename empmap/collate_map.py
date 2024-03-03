from empmap.potential import Potential1D
from empmap.sincdvr import DVR
from empmap.constants import ConstantsManagement
import numpy as np


class EmpiricalMap:
    def __init__(self, file_list=None, file_start=0, file_end=200, file_prefix="Scan_", calc_dir="run_qm/"):
        self.constants = ConstantsManagement()
        if file_list is None:
            self.file_list = np.arange(file_start, file_end)
        else:
            self.file_list = file_list
        self.file_prefix = file_prefix
        self.calc_dir = calc_dir

        self.w01 = []
        self.w12 = []
        self.Eproj = []
        return

    def build_from_dvr(self, dvrs):
        for dvr in dvrs:
            self.w01.append(dvr.w01)
            self.w12.append(dvr.w12)

    def build_base_data(self, **kwargs):
        dvrs, success = self._obtain_dvrs(**kwargs)
        self.build_from_dvr(dvrs)
        self.Eproj = self._obtain_eproj()[success]
        print(np.average(self.w01))
        print(np.average(self.w12))
        return

    def _obtain_dvrs(self, emax=3.0, xmax=1.3, mass1=2.014, mass2=15.999):
        all_dvrs = []
        success = []
        for i, file in enumerate(self.file_list):
            try:
                full_prefix = self.calc_dir + "%d/" % file + self.file_prefix
                pot1d = Potential1D(full_prefix + "rOHs.dat", full_prefix + "energies.dat",
                                    full_prefix + "dipoles.dat", full_prefix + "eOHs.dat")
                pot1d.fit_potential_to_poly(3)
                pot1d.fit_dipole_to_poly(2)
                dvr = DVR(pot1d, emax=emax, xmax=xmax,
                          mass1=mass1, mass2=mass2)
                dvr.do_calculation()
                all_dvrs.append(dvr)
                success.append(True)
            except:
                print("Failed to load DVR for file %d" % file)
                success.append(False)
        success = np.array(success)
        return all_dvrs, success

    def _obtain_eproj(self):
        eproj = []
        for file in self.file_list:
            full_prefix = self.calc_dir + "%d/" % file + self.file_prefix
            eproj.append(np.loadtxt(full_prefix + "proj_field.dat"))
        return np.array(eproj)
