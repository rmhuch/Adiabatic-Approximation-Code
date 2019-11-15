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


