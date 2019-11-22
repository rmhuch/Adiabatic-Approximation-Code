import numpy as np
import os


class Molecule:
    def __init__(self, MoleculeName=None, atom_str=None, method=None, scanCoords=None, **kwargs):
        self.MoleculeName = MoleculeName
        if MoleculeName is None:
            raise Exception("No Molecule to build.")
        self.atom_str = atom_str
        self.params = kwargs
        self.method = method
        if method is None: 
            raise Exception("No method defined.")
        self.scanCoords = scanCoords
        self._mol_dir = None
        self._scanLogs = None
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
        scan_dir = os.path.join(self.mol_dir, "2D Scans", "logs")
        if self.method == "rigid":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*rigid*.log"))))
        elif self.method == "partrig":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*partrig*.log"))))
        elif self.method == "partrel":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*partrel*.log"))))
        elif self.method == "relax":
            allscans = list(sorted(glob.glob(os.path.join(scan_dir, "*relax*.log"))))
        else:
            raise Exception("Weird. I don't know that one.")
        return allscans

    def getMass(self):
        """Uses self.atom_str and Converter.py to create a mass array based off of the molecule. """
        from Converter import Constants
        masses = np.array([Constants.mass(a, to_AU=True) for a in self.atom_str])
        return masses


    # class method perhaps?
    # def atom_dists(coords):
    #     x_pos = (0, 1)
    #     xps = coords[:, x_pos]
    #     xdiffs = np.diff(xps, axis=1)
    #     xdiffs = xdiffs.reshape((len(xdiffs), 3))
    #     xdists = np.linalg.norm(xdiffs, axis=1)
    #     xdists = np.around(xdists, 9)
    #
    #     y_pos = (1, 2)
    #     yps = coords[:, y_pos]
    #     ydiffs = np.diff(yps, axis=1)
    #     ydiffs = ydiffs.reshape((len(ydiffs), 3))
    #     ydists = np.linalg.norm(ydiffs, axis=1)
    #     ydists = np.around(ydists, 9)
    #
    #     oo_pos = (1, 5)
    #     oops = coords[:, oo_pos]
    #     oodiffs = np.diff(oops, axis=1)
    #     oodiffs = oodiffs.reshape((len(oodiffs), 3))
    #     oodists = np.linalg.norm(oodiffs, axis=1)
    #     oodists = np.around(oodists, 9)
    #
    #     oo2_pos = (1, 8)
    #     oo2ps = coords[:, oo2_pos]
    #     oo2diffs = np.diff(oo2ps, axis=1)
    #     oo2diffs = oo2diffs.reshape((len(oo2diffs), 3))
    #     oo2dists = np.linalg.norm(oo2diffs, axis=1)
    #     oo2dists = np.around(oo2dists, 9)
    #
    #     dists = np.column_stack((xdists, ydists, oodists, oo2dists))
    #     return dists

