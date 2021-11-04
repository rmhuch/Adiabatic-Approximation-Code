import numpy as np
import os


class Molecule:
    def __init__(self, MoleculeName=None, atom_str=None, method=None, dimension=None,
                 scanCoords=None, embed_dict=None, OH=False, **kwargs):
        self.MoleculeName = MoleculeName
        if MoleculeName is None:
            raise Exception("No Molecule to build.")
        if self.MoleculeName == "H9O4pls":
            self.OOmin = 2.5696
            self.XHmin = 0.2723
        elif self.MoleculeName == "H7O3pls":
            self.OOmin = 2.5066
            self.XHmin = 0.2179
        else:
            self.OOmin = None
            self.XHmin = None
        self.atom_str = atom_str
        self.params = kwargs
        self.method = method
        if method is None: 
            raise Exception("No method defined.")
        self.dimension = dimension
        self.scanCoords = scanCoords
        self.embed_dict = embed_dict
        self.OH = OH
        self._mol_dir = None
        self._scanLogs = None
        self._logData = None
        self._scanValDict = None
        self._massArray = None

    @property
    def mol_dir(self):
        if self._mol_dir is None:
            self._mol_dir = self.get_mainD()
        return self._mol_dir

    @property
    def scanLogs(self):
        if self._scanLogs is None: 
            self._scanLogs = self.get_2Dlogs()
        return self._scanLogs

    @property
    def logData(self):
        if self._logData is None:
            from GaussianHandler import LogInterpreter
            if self.method == "rigid":
                optBool = False
            else:
                optBool = True
            self._logData = LogInterpreter(*self.scanLogs, moleculeObj=self, optimized=optBool)
        return self._logData

    @property
    def scanValDict(self):
        if self._scanValDict is None:
            full_grid, sc1, sc2 = self.getGrid()
            self._scanValDict = {"fullGrid": full_grid,
                                 "scanCoord1": sc1,
                                 "scanCoord2": sc2}
        return self._scanValDict

    @property
    def massArray(self):
        if self._massArray is None:
            self._massArray = self.getMass()
        return self._massArray

    def get_mainD(self):
        """pulls path to MoleculeName folder in udrive."""
        udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        mainD = os.path.join(udrive, self.MoleculeName)
        return mainD

    def get_2Dlogs(self):
        """For given method, pulls log files from udrive file system."""
        import glob
        if self.OH:
            scan_dir = os.path.join(self.mol_dir, "2D Scans OH")
        else:
            scan_dir = os.path.join(self.mol_dir, "2D Scans XH")
        if self.method == "rigid":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*_rigid_*.log"))))
        elif self.method == "partrig":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*partrig*.log"))))
        elif self.method == "partrel":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*partrel*.log"))))
        elif self.method == "relax":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*_relax_*.log"))))
        else:
            raise Exception("Weird. I don't know that one.")
        return allscans

    def getGrid(self):
        """uses 2D log files to pull the unique values along the scan grid"""
        full_grid = np.array(list(self.logData.cartesians.keys()))
        sc1 = np.sort(np.unique(full_grid[:, 0]))
        sc2 = np.sort(np.unique(full_grid[:, 1]))
        return full_grid, sc1, sc2

    def getMass(self):
        """Uses self.atom_str and Converter.py to create a mass array based off of the molecule. """
        from Converter import Constants
        masses = np.array([Constants.mass(a, to_AU=True) for a in self.atom_str])
        return masses


class MolecularOperations:
    def __init__(self, moleculeObj=None, LogData=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule.embed_dict is None:
            raise Exception("No embedding parameters set.")
        else:
            self.embed_dict = self.molecule.embed_dict
        self.method = self.molecule.method
        self.atom_str = self.molecule.atom_str
        if LogData is not None:
            self.logData = LogData
        else:
            self.logData = moleculeObj.logData
        self._coords = None
        self._embeddedCoords = None
        self._embeddedDips = None
    
    @property 
    def coords(self):
        if self._coords is None:
            self._coords = np.array(list(self.logData.cartesians.values()))
        return self._coords

    @property
    def embeddedCoords(self):
        if self._embeddedCoords is None:
            self._embeddedCoords, self._embeddedDips = self.many_rotations(**self.embed_dict)
        return self._embeddedCoords

    @property
    def embeddedDips(self):
        if self._embeddedDips is None:
            self._embeddedCoords, self._embeddedDips = self.many_rotations(**self.embed_dict)
        return self._embeddedDips

    @staticmethod
    def calculateBonds(coords, atom1, atom2):
        pos = (atom1, atom2)
        ps = coords[:, pos]
        diffs = np.diff(ps, axis=1)
        diffs = diffs.reshape((len(diffs), 3))
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.around(dists, 5)
        return dists 

    @staticmethod
    def get_xyz(filename, coords, atom_str):
        """writes an xyz file to visualize structures from a scan.
            :arg filename: string name of the xyz file to be written
            :returns saves an xyz file of file_name """
        with open(filename, 'w') as f:
            if len(coords.shape) == 2:
                f.write("%s \n structure \n" % (len(atom_str)))
                for j in range(len(atom_str)):
                    f.write("%s %5.8f %5.8f %5.8f \n" %
                            (atom_str[j], coords[j, 0], coords[j, 1], coords[j, 2]))
                f.write("\n")
            else:
                for i in range(len(coords)):
                    f.write("%s \n structure %s \n" % (len(atom_str), (i + 1)))
                    for j in range(len(atom_str)):
                        f.write("%s %5.8f %5.8f %5.8f \n" %
                                (atom_str[j], coords[i, j, 0], coords[i, j, 1], coords[i, j, 2]))
                    f.write("\n")

    @staticmethod
    def rot1(coords, dips, xAxis_atom):
        if xAxis_atom is None:
            raise Exception("No x-axis atom defined")
        # step 1: rotate about z-axis.
        y = coords[:, xAxis_atom, 1]
        x = coords[:, xAxis_atom, 0]
        phi_1 = np.arctan2(y, x)
        cphi = np.cos(phi_1)
        sphi = np.sin(phi_1)
        z_rotator = np.zeros((len(coords), 3, 3))
        z_rotator[:, 0, :] = np.column_stack((cphi, sphi, np.zeros(len(coords))))
        z_rotator[:, 1, :] = np.column_stack((-1 * sphi, cphi, np.zeros(len(coords))))
        z_rotator[:, 2, :] = np.reshape(np.tile([0, 0, 1], len(coords)), (len(coords), 3))
        z_coord = np.matmul(z_rotator, coords.transpose(0, 2, 1)).transpose(0, 2, 1)
        z_dip = np.matmul(z_rotator, dips.transpose(0, 2, 1)).transpose(0, 2, 1)
        # step 2: rotate about y-axis.
        z = z_coord[:, xAxis_atom, 2]
        rho = z_coord[:, xAxis_atom, 0]
        phi_1p = np.arctan2(z, rho)
        cphi_1p = np.cos(phi_1p)
        sphi_1p = np.sin(phi_1p)
        y_rotator = np.zeros((len(z_coord), 3, 3))
        y_rotator[:, 0, :] = np.column_stack((cphi_1p, np.zeros(len(z_coord)), sphi_1p))
        y_rotator[:, 1, :] = np.reshape(np.tile([0, 1, 0], len(z_coord)), (len(z_coord), 3))
        y_rotator[:, 2, :] = np.column_stack((-1 * sphi_1p, np.zeros(len(z_coord)), cphi_1p))
        y_coord = np.matmul(y_rotator, z_coord.transpose(0, 2, 1)).transpose(0, 2, 1)
        y_dip = np.matmul(y_rotator, z_dip.transpose(0, 2, 1)).transpose(0, 2, 1)
        return y_coord, y_dip

    @staticmethod
    def rot2(coords, dips, xyPlane_atom, outerO1, outerO2):
        if xyPlane_atom is not None:
            z5 = coords[:, xyPlane_atom, 2]
            y5 = coords[:, xyPlane_atom, 1]
        elif outerO1 is not None and outerO2 is not None:
            # define bisector of other Os
            o1 = coords[:, outerO1, :]
            nrm1 = np.linalg.norm(o1, axis=1)
            onew = np.zeros((len(coords), 3))
            for i, row in enumerate(o1):
                onew[i, 0] = row[0] / nrm1[i]
                onew[i, 1] = o1[i, 1] / nrm1[i]
                onew[i, 2] = o1[i, 2] / nrm1[i]
            o2 = coords[:, outerO2, :]
            nrm2 = np.linalg.norm(o2, axis=1)
            otwo = np.zeros((len(coords), 3))
            for i, row in enumerate(o1):
                otwo[i, 0] = row[0] / nrm2[i]
                otwo[i, 1] = o2[i, 1] / nrm2[i]
                otwo[i, 2] = o2[i, 2] / nrm2[i]
            bisector = (onew + otwo)
            z5 = bisector[:, 2]
            y5 = bisector[:, 1]
        else:
            raise Exception("rotation to xy-plane not defined")
        phi_3 = np.arctan2(z5, y5)
        cphi_3 = np.cos(phi_3)
        sphi_3 = np.sin(phi_3)
        x_rotator = np.zeros((len(coords), 3, 3))
        x_rotator[:, 0, :] = np.reshape(np.tile([1, 0, 0], len(coords)), (len(coords), 3))
        x_rotator[:, 1, :] = np.column_stack((np.zeros(len(coords)), cphi_3, sphi_3))
        x_rotator[:, 2, :] = np.column_stack((np.zeros(len(coords)), -1 * sphi_3, cphi_3))
        x_coord = np.matmul(x_rotator, coords.transpose(0, 2, 1)).transpose(0, 2, 1)
        x_dip = np.matmul(x_rotator, dips.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x_coord, x_dip

    @staticmethod
    def inverter(coords, dips, inversion_atom):
        coords[:, :, -1] *= np.sign(coords[:, inversion_atom, -1])[:, np.newaxis]
        dips[:, :, -1] *= np.sign(coords[:, inversion_atom, -1])[:, np.newaxis]
        return coords, dips

    def many_rotations(self, centralO_atom=None, xAxis_atom=None, xyPlane_atom=None,
                       outerO1=None, outerO2=None, inversion_atom=None, **params):
        from Converter import Constants
        all_coords = Constants.convert(self.coords, "angstroms", to_AU=True)
        if len(self.logData.logs) == 1:
            dop = self.logData.get_dips(optimized=True)
        else:
            dop = self.logData.dipoles
        all_dips = np.array(list(dop.values()))
        all_dips = all_dips.reshape((len(all_coords), 1, 3))
        if centralO_atom is None:
            raise Exception("No origin atom defined")
        # shift to origin
        o_coords = all_coords - all_coords[:, np.newaxis, centralO_atom]
        o_dips = all_dips - all_coords[:, np.newaxis, centralO_atom]
        # rotation to x-axis
        r1_coords, r1_dips = self.rot1(o_coords, o_dips, xAxis_atom)
        if xyPlane_atom is None and inversion_atom is None:
            # returns coords rotated to x-axis
            rot_coords = r1_coords
            dipadedodas = r1_dips.reshape(len(all_coords), 3)
        elif xyPlane_atom or outerO1 is None and isinstance(inversion_atom, int):
            # returns coords rotated to x-axis and inverted about a designated atom
            rot_coords, rot_dips = self.inverter(r1_coords, r1_dips, inversion_atom)  # inversion of designated atom
            dipadedodas = rot_dips.reshape(len(all_coords), 3)
        elif inversion_atom is None:
            # returns coords rotated to xyplane
            r2_coords, r2_dips = self.rot2(r1_coords, r1_dips, xyPlane_atom, outerO1, outerO2)  # rotation to xy-plane
            rot_coords = r2_coords
            dipadedodas = r2_dips.reshape(len(all_coords), 3)
        else:
            # returns coords rotated to xyplane and inverted about a designated atom
            r2_coords, r2_dips = self.rot2(r1_coords, r1_dips, xyPlane_atom, outerO1, outerO2)  # rotation to xy-plane
            rot_coords, rot_dips = self.inverter(r2_coords, r2_dips, inversion_atom)  # inversion of designated atom
            dipadedodas = rot_dips.reshape(len(all_coords), 3)
        # np.save(os.path.join(self.molecule.mol_dir, "structures", f"FD{self.molecule.MoleculeName}_rotdips2021.npy"),
        #         dipadedodas)
        # np.save(os.path.join(self.molecule.mol_dir, "structures", f"FD{self.molecule.MoleculeName}_rotcoords2021.npy"),
        #         rot_coords)
        # self.get_xyz(f"{self.molecule.MoleculeName}_{self.molecule.method}_rotcoords2021.xyz", rot_coords,
        #              self.atom_str)
        return rot_coords, dipadedodas  # bohr & debye
