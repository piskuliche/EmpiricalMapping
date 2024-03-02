import numpy as np


class GaussianLogFile:
    def __init__(self, logname):
        self.logname = logname
        self.energies = self.grab_energies()
        self.dipole = self.grab_dipole()

    def grab_energies(self):
        energies = []
        with open(self.logname, 'r') as f:
            for line in f:
                if "SCF Done" in line:
                    energy = float(line.split()[4])
                    energies.append(energy)

        if len(energies) == 0:
            raise ValueError("No energies found in the log file")

        return np.array(energies)

    def grab_dipole(self):
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
