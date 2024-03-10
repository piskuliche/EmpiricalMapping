""" Sets up the calculations for the empirical map.

Notes:
------
The MapSetup class is used to set up the calculations for the empirical map.
It is initialized with the selection, bond atoms, central atom, bond masses, inner
cutoff, outer cutoff, calculation directory, scan dr, ngrid, and rmin.

It then uses this information, along with a trajectory, to grab clusters from the
frames and write Gaussian input files for the clusters.

To Do:
------
1) Make Masses + Total Mass Read In + charges
2) Add Selections for Bond to be scanned
3) Refactor _field_on_atom_from_cluster to use MDAnalysis charges.
4) Refactor calc_eOH to use explicit "safer" selection
5) Allow for the user to specify the level of theory for Gaussian
6) Add other QM programs that you can call - ase?
7) Generalize rOH distance calcualtor to all bonds.

"""
__all__ = ["MapSetup"]

import os
import warnings

import numpy as np
import MDAnalysis as mda

from empmap.constants import ConstantsManagement


class MapSetup:
    def __init__(self, calc_dir='newmap/', selection="type O", bond_atoms=[0, 1], central_atom=1, bond_masses=[1.008, 17.007],
                 inner_cutoff=4.0, outer_cutoff=8.0,  scan_dr=0.04, ngrid=14, rmin=0.72, nproc=4, mem=20):
        """
        Initialize the MapSetup class

        Examples:
        ---------
        >>> from empmap.emp_setup import MapSetup
        >>> setup = MapSetup()
        >>> setup.load_universe("water.gro", "water.xtc")
        >>> setup.grab_clusters_from_frames([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> setup.write_gaussian(resid, inner, outer, 0, "scan_")

        Warnings will be issued if the universe doesn't include needed information, such as
        charges, masses, or types during initialization. 

        These warnings will be raised as errors if clusters are tried to be grabbed from the universe
        before these are added to the universe.

        To add charges to the universe, use the set_charges_from_list or set_charges_by_type methods.

        Example:
        --------
        >>> from empmap.emp_setup import MapSetup
        >>> setup = MapSetup()
        >>> setup.load_universe("water.gro", "water.xtc")
        >>> charges = {"O": -0.834, "H": 0.417}
        >>> setup.set_charges_by_type(charges)

        Parameters:
        -----------
        calc_dir : str
            The directory for the calculations. (The default is 'newmap/' which
            will create a new directory called 'newmap' in the current working directory.)
        selection : str
            The MDAnalysis selection string for the molecules around the
            central resid. (The default is "type O" which selects all the oxygens within
            the cutoff distance of the central atom.)
        bond_atoms : list
            The indices of the atoms in the bond. (The default is [0, 1] which
            selects the first two atoms in the central residue. For water, this
            would be the O-H bond.)
        central_atom : int
            The index of the central atom. (The default is 1 which selects the
            second atom in the central residue. For water, this would be the first
            hydrogen atom.)
        bond_masses : list
            The masses of the atoms in the bond. These masses are used
            to shift the bond coordinate while maintaining a
            constant center of mass (The default is [1.008, 17.007]
            which is the mass of H, and the mass of O+H, respectively.)
        inner_cutoff : float
            The inner cutoff for the cluster. This cutoff specifies what should
            be treated quantum mechanically. (The default is 4.0 angstroms.)
        outer_cutoff : float
            The outer cutoff for the cluster. This cutoff specifies what should
            be treated classically. (The default is 8.0 angstroms.)
        scan_dr : float
            The step size for the scan along the bond axis. (The default is 0.04 angstroms.)
        ngrid : int
            The number of gridpoints for the scan. (The default is 14.)
        rmin : float
            The minimum distance for the scan. (The default is 0.72 angstroms.)
        nproc : int
            The number of processors to use for the Gaussian calculations. (The default is 4.)
        mem : int
            The amount of memory to use for the Gaussian calculations. (The default is 20.)

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
        """ Print the description of the MapSetup class

        """
        print("Selection: %s" % self.selection)
        print("Inner cutoff: %10.5f" % self.inner_cutoff)
        print("Outer cutoff: %10.5f" % self.outer_cutoff)
        return

    def load_universe(self, *args, **kwargs):
        """
        Load the universe using the topology and trajectory files

        Stores the MDAnalysis universe as an attribute of the class.

        Parameters:
        -----------
        args: 
            The topology and trajectory files, respectively. These are
            passed directly to the MDAnalysis Universe class.
        kwargs:
            Any keyword arguments to pass to the MDAnalysis Universe class.
            Please see the MDAnalysis documentation for more information.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError: 
            If the universe cannot be loaded.

        """
        if len(args) == 0:
            raise ValueError("Please provide something to build the universe.")

        try:
            self.universe = mda.Universe(*args, **kwargs)
        except:
            raise ValueError("Could not load the universe.")

        self._test_universe()

        return

    def set_attribute_from_list(self, attribute, inputdata):
        """
        Set the charges from a list

        Parameters:
        -----------
        attribute : str
            The attribute to set. This should be a string that is a valid
            attribute of the atoms in the universe. [e.g. "charges"]
        inputdata : array_like
            The inputdata for the atoms in the universe. These should be in the
            same order as the atoms in the universe.

        Returns:
        --------
        None

        Raises:
        -------
        TypeError
            If the attribute is not a string.
        ValueError
            If the number of charges does not match the number of atoms in the universe.

        """
        if not isinstance(attribute, str):
            raise TypeError("The attribute must be a string.")

        if len(inputdata) != len(self.universe.atoms):
            raise ValueError(
                "The number of inputdata does not match the number of atoms in the universe.")

        self.universe.add_TopologyAttr(attribute)
        setattr(self.universe.atoms, attribute, inputdata)
        return

    def set_attribute_by_type(self, attribute, input_dict):
        """
        Set the inputdata by type using a dictionary

        Parameters:
        -----------
        attribute : str
            The attribute to set. This should be a string that is a valid
            attribute of the atoms in the universe. [e.g. "charges"]
        input_dict : dict
            The input data dictionary mapped to types. The keys should be the
            types in the universe, and the values should be the input data for
            the types. (e.g. {"O": 0.834, "H": 0.417})

        Returns:
        --------
        None

        Raises:
        -------
        TypeError
            If the input data is not a dictionary
        ValueError
            If the number of charges does not match the number of atoms in the universe.

        """
        if not isinstance(input_dict, dict):
            raise TypeError("The input data must be a dictionary.")
        try:
            self.universe.atoms.types
        except:
            raise ValueError("No types found in the universe.")

        # Build the data list
        data_list = []
        for type in self.universe.atoms.types:
            data_list.append(input_dict[type])
        # Set the attribute
        self.set_attribute_from_list(attribute, data_list)
        return

    def grab_clusters_from_frames(self, frames, file_prefix="map_"):
        """ Grab the clusters from the frames

        Notes:
        ------
        This method grabs the clusters from the frames in the trajectory. It
        then writes the Gaussian input files for the clusters. It also calculates
        the static parameters for the clusters and writes them to files.

        Parameters:
        -----------
        frames : list
            The frames to grab the clusters from. These should be the indices
            of the frames in the trajectory. (e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        file_prefix : str
            The file prefix for the Gaussian input files. (The default is map)
            Thus, this will create files like map_0.gjf, map_1.gjf, etc.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError
            If the universe is not set up properly. This is based on the status
            of the universe set by the _test_universe method.

        """
        status = self._test_universe()
        if not status:
            raise ValueError("Universe is not set up properly.")

        # Loop over the frames in the trajecotry
        for ts in self.universe.trajectory[frames]:
            # Grab the box dimensions
            box = ts.dimensions[:3]
            # Grab the clusters
            res, inner, outer = self._grab_cluster(box)
            # Write the Gaussian input files
            vib_bond_distances = self.write_gaussian(
                res, inner, outer, ts.frame, file_prefix)
            self.write_output_data(
                res, inner, outer, ts.frame, file_prefix, vib_bond_distances)

        return

    def write_output_data(self, resid, inner, outer, frame_number, file_prefix, vib_bond_distances):
        """ Write the output data needed for the empirical map

        Notes:
        ------
        This method writes the output data needed for the empirical map. This
        includes the bond vector, the field on the residue from the cluster, the
        projected field on the residue from the cluster, and the distance from the
        gridpoint.

        The files created (for an example directory) are:
        newmap/0/scan_eOHs.dat          # The bond vector values for the cluster (eOH)
        newmap/0/scan_fields.dat        # The field on the residue from the cluster
        newmap/0/scan_rOHs.dat          # The rOH values for the scan
        newmap/0/scan_proj_field.dat    # The projected field on the residue from the cluster

        Parameters:
        -----------
        resid : MDAnalysis.AtomGroup
            The residue to write the output data for. This is the central residue.
        inner : MDAnalysis.AtomGroup
            The inner cluster to write the output data for. This is the cluster
            treated quantum mechanically.
        outer : MDAnalysis.AtomGroup
            The outer cluster to write the output data for. This is the cluster
            treated classically.
        frame_number : int
            The frame number for the output data. This is used to create a
            subdirectory for the frame.
        file_prefix : str
            The file prefix for the output data. This is used to create the
            file names for the output data.
        vib_bond_distances : np.ndarray, shape(self.ngrid)
            The bond distance values for the scan.

        Returns:
        --------
        None

        """

        calc_subdir = self.calc_dir + "%d/" % frame_number

        # Calculate the Static Parameters
        # calculate the bond vector
        vib_bond_vector = self._calc_bond_vector(
            resid[self.bond_atoms[1]].position,
            resid[self.bond_atoms[0]].position)
        field = self._field_on_atom_from_cluster(resid, inner, outer)
        proj_field = self._project_field(field, vib_bond_vector)

        # Write the Static Parameters to Files
        np.savetxt(calc_subdir+file_prefix+"eOHs.dat", vib_bond_vector)
        np.savetxt(calc_subdir+file_prefix+"fields.dat", field)
        np.savetxt(calc_subdir+file_prefix+"rOHs.dat", vib_bond_distances)
        np.savetxt(calc_subdir+file_prefix+"proj_field.dat", [proj_field])

    def write_gaussian(self, resid, inner, outer, frame_number, file_prefix, functional="b3lyp", basis="6-311G(d,p)"):
        """
        Write the Gaussian input files, and the output data

        Notes:
        ------
        This method writes the Gaussian input files for the clusters. It also
        calculates the static parameters for the clusters and writes them to
        files.

        The files created (for an example directory) are:
        newmap/0/scan_0.gjf             # The Gaussian input file
        newmap/0/scan_0.xyz             # The XYZ file for the cluster (shows the stretched coordinate)


        Parameters:
        -----------
        resid : MDAnalysis.AtomGroup
            The residue to write the Gaussian input files for. This is the
            central residue.
        inner : MDAnalysis.AtomGroup
            The inner cluster to write the Gaussian input files for. This is
            the cluster treated quantum mechanically.
        outer : MDAnalysis.AtomGroup
            The outer cluster to write the Gaussian input files for. This is
            the cluster treated classically.
        frame_number : int
            The frame number for the Gaussian input files. This is used to
            create a subdirectory for the frame.
        file_prefix : str
            The file prefix for the Gaussian input files. This is used to
            create the file names for the Gaussian input files and the
            output data.
        functional : str
            The functional for the Gaussian input files. (The default is "b3lyp")
        basis : str
            The basis set for the Gaussian input files. (The default is "6-311G(d,p)")

        Returns:
        --------
        vib_bond_distances : np.ndarray, shape(self.ngrid)
            The bond distance values for the scan.

        """
        calc_subdir = self.calc_dir + "%d/" % frame_number
        if not os.path.exists(calc_subdir):
            os.makedirs(calc_subdir)

        # Open the gaussian
        finp = open(calc_subdir+file_prefix+"%s.gjf" % frame_number, "w")
        fxyz = open(calc_subdir+file_prefix+"%s.xyz" % frame_number, "w")
        # Write the header of the gaussian file
        finp.write("%"+"NProcShared=%d\n" % self.nproc)
        finp.write("%"+"Mem=%dGB\n" % self.mem)
        finp.write("%chk=calc.chk\n")

        vib_bond_distances = np.zeros(self.ngrid)

        # Loop over the gridpoints
        for n in range(self.ngrid):
            # Add the link block if n>0
            if n > 0:
                finp.write("--Link1--\n")
                finp.write("\n")
            # Write the gridpoint to the file
            vib_bond_distance, atypes, coords = self._write_gridpoint(
                finp, n, resid, inner, outer,
                functional=functional, basis=basis)
            self._write_xyz_traj(fxyz, atypes, coords)
            # Store the distance from the gridpoint
            vib_bond_distances[n] = vib_bond_distance

        # Close the files
        finp.close()
        fxyz.close()

        return vib_bond_distances

    def _write_xyz_traj(self, f, atypes, coords):
        """ Write the XYZ trajectory

        Parameters:
        -----------
        f : file_object
            The file object to write to.
        atypes : list
            The atom types
        coords : np.ndarray, shape(n, 3)
            The coordinates of the atoms

        """
        f.write("%d\n" % len(atypes))
        f.write("Bond Scan\n")
        for i, atype in enumerate(atypes):
            f.write("%s %10.5f %10.5f %10.5f\n" %
                    (atype, coords[i][0], coords[i][1], coords[i][2]))
        return

    def _write_gridpoint(self, f, n, resid, inner, outer, functional="b3lyp", basis="6-311G(d,p)"):
        """ Write the gridpoint to the Gaussian input file

        Parameters:
        -----------
        f : file_object
            The file object to write to.
        n : int
            The gridpoint number.
        resid : MDAnalysis.AtomGroup
            The residue to write the gridpoint for. This is the central residue.
        inner : MDAnalysis.AtomGroup
            The inner cluster to write the gridpoint for. This is the cluster
            treated quantum mechanically.
        outer : MDAnalysis.AtomGroup
            The outer cluster to write the gridpoint for. This is the cluster
            treated classically.
        functional : str
            The functional for the Gaussian input files. (The default is "b3lyp")
        basis : str
            The basis set for the Gaussian input files. (The default is "6-311G(d,p)")

        Returns:
        --------
        bond_distance : float
            The distance from the gridpoint
        atypes : list
            The atom types
        coords : np.ndarray, shape(n, 3)
            The coordinates of the atoms

        """
        bond_distance, rtmp1, rtmp2, rtmpO = self._calc_rOH_distance(resid, n)
        atypes, coords = [], []
        f.write("#n %s/%s empiricaldispersion=gd3 Charge Polar\n" %
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
        return bond_distance, atypes, coords

    def _calc_bond_vector(self, r1, r2):
        """ Calculate the bond vector

        Parameters:
        -----------
        r1 : np.ndarray, shape(3)
            The position of the first atom
        r2 : np.ndarray, shape(3)
            The position of the second atom

        Returns:
        --------
        e_vector : np.ndarray, shape(3)
            The unit vector pointing along the bond

        """
        e_vector = np.zeros(3)
        e_vector = np.subtract(r1, r2)
        norm = np.sqrt(np.dot(e_vector, e_vector))
        e_vector = e_vector / norm
        return e_vector

    def _project_field(self, field, bond_vector):
        """ Project the field onto the bond axis

        Notes:
        ------
        This method projects the field onto the bond axis. This is used to
        calculate the field on the bond axis.

        Parameters:
        -----------
        field : np.ndarray, shape(3)
            The field on the residue from the cluster
        bond_vector : np.ndarray, shape(3)
            The bond unit vector

        Returns:
        --------
        np.ndarray, shape(3)
            The projected field onto the unit vector

        Raises:
        -------
        ValueError
            If the bond vector is not a unit vector.

        """
        if np.abs(np.dot(bond_vector, bond_vector) - 1.0) > 1e-6:
            raise ValueError("Bond vector is not a unit vector.")

        return np.array(np.dot(field, bond_vector)*self.constants.angperau**2.)

    def _calc_rOH_distance(self, resid, n):
        """ Calculate the rOH distance

        Parameters:
        -----------
        resid : MDAnalysis.AtomGroup
            The residue to calculate the rOH distance for. This is the central
            residue.
        n : int
            The gridpoint number.

        Returns:
        --------
            rOH (float): The distance
            rtmp1 (np.ndarray): The position of atom 1
            rtmp2 (np.ndarray): The position of atom 2
            rtmpO (np.ndarray): The position of the oxygen


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

        # Calculate the New Positions for the scan
        # while maintaining the center of mass
        rtmp1 = ratom1 + (self.bond_masses[1]) * \
            e_vector * (self.rmin+self.scan_dr*n)/self.total_mass
        rtmp2 = rother - (self.bond_masses[0])*e_vector * \
            (self.rmin+self.scan_dr*n)/self.total_mass
        rtmpO = ratom1 - (self.bond_masses[0])*e_vector * \
            (self.rmin+self.scan_dr*n)/self.total_mass

        # Calculate the rOH distance
        rOH = np.sqrt(np.sum(np.subtract(rtmp1, rtmpO)**2.))

        return rOH, rtmp1, rtmp2, rtmpO

    def _field_on_atom_from_cluster(self, resid, inner, outer):
        """
        Calculate the field on the residue from the cluster

        Notes:
        ------
        This method calculates the field on the residue from the cluster. This
        is done by summing the field from each atom in the cluster. The field
        from each atom is calculated as:

        F = q(r-r0)/|r-r0|^3

        where q is the charge, r is the position of the atom, and r0 is the
        position of the residue.

        Parameters:
        -----------
        resid : MDAnalysis.AtomGroup
            The residue to calculate the field for. This is the central residue.
        inner : MDAnalysis.AtomGroup
            The inner cluster to calculate the field for. This is the cluster
            treated quantum mechanically.
        outer : MDAnalysis.AtomGroup
            The outer cluster to calculate the field for. This is the cluster
            treated classically.

        Returns:
        --------
        field : np.ndarray, shape(3)
            The field on the residue from the cluster

        """
        cluster = inner.concatenate(outer)

        field = np.zeros(3)

        for atom in cluster:
            dr = resid[self.bond_atoms[1]].position - atom.position
            dist = np.sqrt(np.sum(dr**2.))
            field += atom.charge*dr/dist**3.

        return field

    def _grab_cluster(self, Lbox, resid_override=None):
        """
        Grab the clusters for the current frame

        Notes:
        ------
        This method grabs the clusters for the current frame. It does this by first
        selecting the residue, and then selecting the inner and outer clusters.

        Note - the clusters are wrapped to the central molecule.



        Parameters:
        -----------
        Lbox : np.ndarray, shape(3)
            The box dimensions, in angstroms
        resid_override : int
            The residue to override the random choice with, if desired.

        Returns:
        --------
        resid : MDAnalysis.AtomGroup
            The residue for the current frame
        inner : MDAnalysis.AtomGroup
            The inner cluster for the current frame
        outer : MDAnalysis.AtomGroup
            The outer cluster for the current frame

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
        Wrap the cluster to the central residue

        Notes:
        ------
        This method wraps the cluster to the central residue. It does this by
        calculating the distance from the central residue, and then wrapping the
        cluster to the central residue.

        Parameters:
        -----------
        resid : MDAnalysis.AtomGroup
            The central residue
        towrap : MDAnalysis.AtomGroup
            The cluster to wrap
        Lbox : np.ndarray, shape(3)
            The box dimensions, in angstroms

        Returns:
        --------
        towrap : MDAnalysis.AtomGroup
            The wrapped cluster

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

        Parameters:
        -----------
        None

        Returns:
        --------
        status_of_universe : bool
            The status of the universe. This is True if the universe contains
            the needed information. This includes charges, masses, and types.

        Raises:
        -------
        UserWarning
            If the universe does not contain the needed information. This
            includes charges, masses, and types.

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
