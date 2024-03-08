import numpy as np
import MDAnalysis as mda
import os
import warnings
from empmap.constants import ConstantsManagement


class MapSetup:
    def __init__(self, calc_dir='newmap/', selection="type O", bond_atoms=[0, 1], central_atom=1, bond_masses=[1.008, 17.007],
                 inner_cutoff=4.0, outer_cutoff=8.0,  scan_dr=0.04, ngrid=14, rmin=0.72, nproc=4, mem=20):
        """
        Initialize the MapSetup class

        Args:
            selection (str): The selection string [default: type O]
            center_selection (str): The center selection string [default: type H]
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
            3) Refactor _field_on_atom_from_cluster to use MDAnalysis charges.
            4) Refactor calc_eOH to use explicit "safer" selection
            5) Allow for the user to specify the level of theory for Gaussian
            6) Add other QM programs that you can call - ase?

        """
        print("MapSetup Initializing...")
        print("WARNINGS: ")
        print("1) This code works on water, only. ")
        self.nproc = nproc
        self.mem = mem
        self.selection = selection
        self.bond_atoms = bond_atoms
        self.central_atom = central_atom
        self.inner_cutoff = inner_cutoff
        self.outer_cutoff = outer_cutoff
        self.calc_dir = calc_dir
        self.scan_dr = scan_dr
        self.ngrid = ngrid
        self.rmin = rmin
        # Set the masses
        self.bond_masses = bond_masses
        self.total_mass = self.bond_masses[0]+self.bond_masses[1]
        self.constants = ConstantsManagement()
        if not os.path.exists(self.calc_dir):
            os.makedirs(self.calc_dir)
        return

    def description(self):
        print("Selection: %s" % self.selection)
        print("Inner cutoff: %10.5f" % self.inner_cutoff)
        print("Outer cutoff: %10.5f" % self.outer_cutoff)
        return

    def load_universe(self, *args, **kwargs):
        """
        Load the universe using the topology and trajectory files

        Args:
            topology (str): The topology file
            trajectory (str): The trajectory file

        Returns:
            None

        """
        if len(args) == 0:
            raise ValueError("Please provide something to build the universe.")

        try:
            self.universe = mda.Universe(*args, **kwargs)
        except:
            raise ValueError("Could not load the universe.")

        self._test_universe()

        return

    def set_charges_from_list(self, charges):
        """
        Set the charges

        Args:
            charges (list): The charges

        Returns:
            None
        """
        self.universe.add_TopologyAttr("charges")
        self.universe.atoms.charges = charges
        return

    def set_charges_by_type(self, charge_dict):
        """
        Set the charges by type

        Args:
            charges (dict): The charges

        Returns:
            None
        """
        charges = []
        for type in self.universe.atoms.types:
            charges.append(charge_dict[type])
        self.set_charges_from_list(charges)
        return

    def grab_clusters_from_frames(self, frames):
        """ Grab the clusters from the frames

        Args:
            frames (list): List of frames to grab the clusters from

        Returns:
            None

        """
        status = self._test_universe()
        if not status:
            raise ValueError("Universe is not set up properly.")
        for ts in self.universe.trajectory[frames]:
            box = ts.dimensions[:3]
            res, inner, outer = self._grab_cluster(box)
            self.write_gaussian(res, inner, outer, ts.frame, "scan_")
        return

    def write_gaussian(self, resid, inner, outer, frame_number, file_prefix, functional="b3lyp", basis="6-311G(d,p)"):
        """
        Write the cluster to a file

        Args:
            resid (MDAnalysis.AtomGroup): The residue
            inner (MDAnalysis.AtomGroup): The inner cluster
            outer (MDAnalysis.AtomGroup): The outer cluster
            frame_number (int): The frame number
            file_prefix (str): The file prefix
            functional (str): The functional [default: b3lyp]
            basis (str): The basis set [default: 6-311G(d,p)]

        Returns:
            None

        """
        calc_subdir = self.calc_dir + "%d/" % frame_number
        if not os.path.exists(calc_subdir):
            os.makedirs(calc_subdir)

        finp = open(calc_subdir+file_prefix+"%s.gjf" % frame_number, "w")
        fxyz = open(calc_subdir+file_prefix+"%s.xyz" % frame_number, "w")
        finp.write("%"+"NProcShared=%d\n" % self.nproc)
        finp.write("%"+"Mem=%dGB\n" % self.mem)
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
        eOH = self._calc_bond_vector(
            resid[self.bond_atoms[1]].position,
            resid[self.bond_atoms[0]].position)
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

    def _write_gridpoint(self, f, n, resid, inner, outer, functional="b3lyp", basis="6-311G(d,p)"):
        rOH, rtmp1, rtmp2, rtmpO = self._calc_rOH_distance(resid, n)
        atypes, coords = [], []
        f.write("#n %s/%s empiricaldispersion=gd3 Charge\n" %
                (functional, basis))
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

        for i, pos in enumerate(inner.positions):
            f.write("%s %10.5f %10.5f %10.5f\n" %
                    (inner.atoms.types[i], pos[0], pos[1], pos[2]))
            coords.append(pos)
            atypes.append(inner.atoms.types[i])
        f.write("\n")

        for i, pos in enumerate(outer.positions):
            f.write("%10.5f %10.5f %10.5f %10.5f\n" %
                    (pos[0], pos[1], pos[2], outer.atoms.charges[i]))
            coords.append(pos)
            atypes.append(outer.atoms.types[i])
        f.write("\n")
        return rOH, atypes, coords

    def _calc_bond_vector(self, r1, r2):
        e_vector = np.zeros(3)
        e_vector = np.subtract(r1, r2)
        norm = np.sqrt(np.dot(e_vector, e_vector))
        e_vector = e_vector / norm
        return e_vector

    def _project_field(self, field, eOH):
        return np.array(np.dot(field, eOH)*self.constants.angperau**2.)

    def _calc_rOH_distance(self, resid, n):
        """ Calculate the rOH distance

        Args:
            resid (MDAnalysis.AtomGroup): The residue
            n (int): The gridpoint

        Returns:
            rOH (float): The distance
            rtmp1 (np.ndarray): The position of atom 1
            rtmp2 (np.ndarray): The position of atom 2
            rtmpO (np.ndarray): The position of the oxygen

        Todo:
            Need to make this more general
        """

        # Grab the Positions

        ratom1 = resid[self.bond_atoms[0]].position
        ratom2 = resid[self.bond_atoms[1]].position
        rother = resid[2].position

        # Calculate the Bond Vector (note - H - O, not O - H)
        e_vector = self._calc_bond_vector(ratom2, ratom1)

        rtmp1 = np.zeros(3)
        rtmp2 = np.zeros(3)
        rtmpO = np.zeros(3)

        rtmp1 = ratom1 + (self.bond_masses[1]) * \
            e_vector * (self.rmin+self.scan_dr*n)/self.total_mass
        rtmp2 = rother - (self.bond_masses[0])*e_vector * \
            (self.rmin+self.scan_dr*n)/self.total_mass
        rtmpO = ratom1 - (self.bond_masses[0])*e_vector * \
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

        for atom in cluster:
            dr = resid[self.bond_atoms[1]].position - atom.position
            dist = np.sqrt(np.sum(dr**2.))
            field += atom.charge*dr/dist**3.
        """
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
        """

        return field

    def _grab_cluster(self, Lbox, resid_override=None):
        """
        Grab the clusters for the current frame

        Returns:
            resid (MDAnalysis.AtomGroup): The random residue number
            inner (MDAnalysis.AtomGroup): The inner cluster
            outer (MDAnalysis.AtomGroup): The outer cluster


        TODO:
            1) Make resids be of a certain type - otherwise, this won't work for mixed systems
            2) Need to modify the selections so that it picks the right central atom
                Ideally, this needs to happen in the selection string so that it can happen
                in one step. 
            3) 

        """
        # Pull a Random Resid
        if resid_override is not None:
            random_resid = resid_override
        else:
            random_resid = np.random.choice(self.universe.residues.resids)

        resid_sel = "resid %d" % random_resid
        resid = self.universe.select_atoms("%s" % resid_sel)
        resid_central_index = resid[self.central_atom].index
        # Select the Inner and Outer Clusters
        inner_sel = "same residue as (%s and around %10.5f index %d)" % (
            self.selection, self.inner_cutoff, resid_central_index)
        outer_sel = "same residue as (%s and around %10.5f index %d)" % (
            self.selection, self.outer_cutoff, resid_central_index)

        # Do the Selections

        inner = self.universe.select_atoms("%s" % inner_sel)
        all_outer = self.universe.select_atoms("%s" % outer_sel)
        outer = all_outer.subtract(inner)
        inner = inner.subtract(resid)

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

    def _test_universe(self):
        """
        Test the universe

        Returns:
            None
        """
        status_of_universe = True
        try:
            self.universe.atoms.types
        except:
            warnings.warn("No types found in the universe.", UserWarning)
            status_of_universe = False
        try:
            self.universe.atoms.charges
        except:
            warnings.warn("No charges found in the universe.", UserWarning)
            status_of_universe = False
        try:
            self.universe.atoms.masses
        except:
            warnings.warn("No masses found in the universe.", UserWarning)
            status_of_universe = False

        return status_of_universe


if __name__ == "__main__":
    print("This is a test file. Run test.py instead.")
