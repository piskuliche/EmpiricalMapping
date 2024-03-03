import numpy as np
import MDAnalysis as mda
import os
from empmap.constants import ConstantsManagement


class MapSetup:
    def __init__(self, nmols, selection, inner_cutoff, outer_cutoff, calc_dir='newmap/', scan_dr=0.04, ngrid=14, rmin=0.72):
        """
        Initialize the MapSetup class

        Args:
            nmols (int): The number of molecules
            selection (str): The selection string
            inner_cutoff (float): The inner cutoff
            outer_cutoff (float): The outer cutoff
            calc_dir (str): The calculation directory
            scan_dr (float): The scan dr
            ngrid (int): The number of grid points
            rmin (float): The minimum distance

        Returns:
            None

        To Do:
            1) Make Masses + Total Mass Read In + charges
            2) Add Selections for Bond to be scanned

        """
        print("MapSetup Initializing...")
        print("WARNINGS: ")
        print("1) This code works on water, only. ")
        self.nmols = nmols
        self.selection = selection
        self.inner_cutoff = inner_cutoff
        self.outer_cutoff = outer_cutoff
        self.calc_dir = calc_dir
        self.scan_dr = scan_dr
        self.ngrid = ngrid
        self.rmin = rmin
        self.masses = {"O": 15.999, "H": 1.008, "D": 2.014}
        self.charges = {"O": -0.834, "H": 0.417, "D": 0.417}
        self.total_mass = self.masses["O"] + \
            self.masses["H"] + self.masses["D"]
        self.constants = ConstantsManagement()
        if not os.path.exists(self.calc_dir):
            os.makedirs(self.calc_dir)
        return

    def description(self):
        print("Number of molecules: %d" % self.nmols)
        print("Selection: %s" % self.selection)
        print("Inner cutoff: %10.5f" % self.inner_cutoff)
        print("Outer cutoff: %10.5f" % self.outer_cutoff)
        return

    def load_universe(self, topology=None, trajectory=None):
        """
        Load the universe using the topology and trajectory files

        Args:
            topology (str): The topology file
            trajectory (str): The trajectory file

        Returns:
            None

        """
        if trajectory is None:
            raise ValueError("Trajectory file not found")
        if topology is None:
            try:
                self.universe = mda.Universe(trajectory)
            except:
                raise
        else:
            try:
                self.universe = mda.Universe(topology, trajectory)
            except:
                raise
        return

    def grab_clusters_from_frames(self, frames):
        """ Grab the clusters from the frames

        Args:
            frames (list): List of frames to grab the clusters from

        Returns:
            None

        """
        for ts in self.universe.trajectory[frames]:
            box = ts.dimensions[:3]
            res, inner, outer = self._grab_cluster(box)
            self.write_gaussian(res, inner, outer, ts.frame, "scan_")
        return

    def write_gaussian(self, resid, inner, outer, frame_number, file_prefix):
        """
        Write the cluster to a file

        Args:
            cluster (MDAnalysis.AtomGroup): The cluster to write
            file_prefix (str): The filename to write to

        Returns:
            None

        """
        calc_subdir = self.calc_dir + "%d/" % frame_number
        if not os.path.exists(calc_subdir):
            os.makedirs(calc_subdir)

        finp = open(calc_subdir+file_prefix+"%s.gjf" % frame_number, "w")
        fxyz = open(calc_subdir+file_prefix+"%s.xyz" % frame_number, "w")
        finp.write("%NProcShared=28\n")
        finp.write("%Mem=100GB\n")
        finp.write("%chk=calc.chk\n")
        rOHs = np.zeros(self.ngrid)
        for n in range(self.ngrid):
            if n > 0:
                finp.write("--Link1--\n")
                finp.write("\n")
            # Write the gridpoint to the file
            rOH, atypes, coords = self._write_gridpoint(
                finp, n, resid, inner, outer)
            self._write_xyz_traj(fxyz, atypes, coords)
            # Store the distance from the gridpoint
            rOHs[n] = rOH
        finp.close()
        fxyz.close()

        # Calculate the Static Parameters
        eOH = self._calc_eOH(resid)
        field = self._field_on_atom_from_cluster(resid, inner, outer)
        proj_field = self._project_field(field, eOH)

        np.savetxt(calc_subdir+file_prefix+"eOHs.dat", eOH)
        np.savetxt(calc_subdir+file_prefix+"fields.dat", field)
        np.savetxt(calc_subdir+file_prefix+"rOHs.dat", rOHs)
        np.savetxt(calc_subdir+file_prefix+"proj_field.dat", [proj_field])
        return

    def _write_xyz_traj(self, f, atypes, coords):
        f.write("%d\n" % len(atypes))
        f.write("Water Scan\n")
        for i, atype in enumerate(atypes):
            f.write("%s %10.5f %10.5f %10.5f\n" %
                    (atype, coords[i][0], coords[i][1], coords[i][2]))
        return

    def _write_gridpoint(self, f, n, resid, inner, outer):
        rOH, rtmp1, rtmp2, rtmpO = self._calc_rOH_distance(resid, n)
        rinner = inner.positions
        intypes = inner.atoms.types
        router = outer.positions
        outtypes = outer.atoms.types
        atypes, coords = [], []
        f.write("#n B3LYP/6-311G(d,p) empiricaldispersion=gd3 Charge\n")
        f.write("\n")
        f.write("Water Scan\n")
        f.write("\n")
        f.write("0 1\n")
        f.write("O %10.5f %10.5f %10.5f\n" % (rtmpO[0], rtmpO[1], rtmpO[2]))
        f.write("H %10.5f %10.5f %10.5f\n" % (rtmp1[0], rtmp1[1], rtmp1[2]))
        f.write("H %10.5f %10.5f %10.5f\n" % (rtmp2[0], rtmp2[1], rtmp2[2]))
        coords.append(rtmpO)
        coords.append(rtmp1)
        coords.append(rtmp2)
        atypes.append("O")
        atypes.append("H")
        atypes.append("H")
        for i, pos in enumerate(rinner):
            f.write("%s %10.5f %10.5f %10.5f\n" %
                    (intypes[i], pos[0], pos[1], pos[2]))
            coords.append(pos)
            atypes.append(intypes[i])
        f.write("\n")
        for i, pos in enumerate(router):
            f.write("%10.5f %10.5f %10.5f %10.5f\n" %
                    (pos[0], pos[1], pos[2], self.charges[outtypes[i]]))
            coords.append(pos)
            atypes.append(outtypes[i])
        f.write("\n")
        return rOH, atypes, coords

    def _calc_eOH(self, resid):
        eOH = np.zeros(3)
        rO = resid.select_atoms("type O").positions
        rH = resid.select_atoms("type H").positions
        eOH = np.subtract(rH[0], rO[0])
        norm = np.sqrt(np.dot(eOH, eOH))
        eOH = eOH / norm
        return eOH

    def _project_field(self, field, eOH):
        return np.array(np.dot(field, eOH)*self.constants.angperau**2.)

    def _calc_rOH_distance(self, resid, n):
        eOH = self._calc_eOH(resid)
        rO = resid.select_atoms("type O").positions[0]
        rH2 = resid.select_atoms("type H").positions[1]
        rtmp1 = np.zeros(3)
        rtmp2 = np.zeros(3)
        rtmpO = np.zeros(3)
        rtmp1 = rO + (self.masses["O"] + self.masses["D"]) * \
            eOH * (self.rmin+self.scan_dr*n)/self.total_mass
        rtmp2 = rH2 - (self.masses["H"])*eOH * \
            (self.rmin+self.scan_dr*n)/self.total_mass
        rtmpO = rO - (self.masses["H"])*eOH * \
            (self.rmin+self.scan_dr*n)/self.total_mass
        rOH = np.sqrt(np.sum(np.subtract(rtmp1, rtmpO)**2.))
        return rOH, rtmp1, rtmp2, rtmpO

    def _field_on_atom_from_cluster(self, resid, inner, outer):
        """
        Calculate the field on the residue from the cluster

        Args:
            resid (MDAnalysis.AtomGroup): The residue
            inner (MDAnalysis.AtomGroup): The inner cluster
            outer (MDAnalysis.AtomGroup): The outer cluster

        Returns:
            field (np.ndarray): The field on the residue

        """
        cluster = inner.concatenate(outer)

        field = np.zeros(3)
        resO = resid.select_atoms("type O").positions[0]
        resH = resid.select_atoms("type H").positions[0]

        rO = cluster.select_atoms("type O").positions
        rH = cluster.select_atoms("type H").positions

        for pos in rO:
            # HO
            drHO = resH - pos
            rHO = np.sqrt(np.sum(drHO**2.))
            field += self.charges["O"]*drHO/rHO**3
        for pos in rH:
            drHH = resO - pos
            rHH = np.sqrt(np.sum(drHH**2.))
            field += self.charges["H"]*drHH/rHH**3

        return field

    def _grab_cluster(self, Lbox, resid_override=None):
        """
        Grab the clusters for the current frame

        Returns:
            resid (MDAnalysis.AtomGroup): The random residue number
            inner (MDAnalysis.AtomGroup): The inner cluster
            outer (MDAnalysis.AtomGroup): The outer cluster

        """
        # Pull a Random Resid
        if resid_override is not None:
            random_resid = resid_override
        else:
            random_resid = np.random.choice(self.universe.residues.resids)

        # Select the Inner and Outer Clusters
        resid_sel = "resid %d" % random_resid
        inner_sel = "same residue as (%s and around %10.5f resid %d)" % (
            self.selection, self.inner_cutoff, random_resid)
        outer_sel = "same residue as (%s and around %10.5f resid %d)" % (
            self.selection, self.outer_cutoff, random_resid)

        # Do the Selections
        resid = self.universe.select_atoms("%s" % resid_sel)
        inner = self.universe.select_atoms("%s" % inner_sel)
        all_outer = self.universe.select_atoms("%s" % outer_sel)
        outer = all_outer.subtract(inner)
        inner = inner.subtract(resid)

        all_group = resid.concatenate(inner).concatenate(outer)
        com = all_group.center_of_mass()

        # Translate the Clusters
        # This little section of code ensures that the droplets are centered in the box (which goes from 0 to Lbox)
        # Importantly, it also wraps the atoms around the droplet, so you don't end up with the droplet split
        # across the box. Some care needs to be taken here - you don't want your droplet to be larger
        # than the box. If that happens, there will be a problem.
        resid = self._wrap_to_resid(resid, resid, Lbox)
        inner = self._wrap_to_resid(resid, inner, Lbox)
        outer = self._wrap_to_resid(resid, outer, Lbox)

        return resid, inner, outer

    def _wrap_to_resid(self, resid, towrap, Lbox):
        """
        Wrap the cluster to the box

        Args:
            resid (MDAnalysis.AtomGroup): The residue
            towrap (MDAnalysis.AtomGroup): The cluster to wrap
            Lbox (np.ndarray): The box dimensions

        Returns:
            towrap (MDAnalysis.AtomGroup): The wrapped cluster
        """
        central_pos = resid.positions[0]
        newpos = []
        for pos in towrap.positions:
            dr = pos - central_pos
            dr = dr - Lbox*np.round(dr/Lbox)
            newpos.append(dr+central_pos)
        towrap.positions = np.array(newpos)
        return towrap


if __name__ == "__main__":
    print("This is a test file. Run test.py instead.")
