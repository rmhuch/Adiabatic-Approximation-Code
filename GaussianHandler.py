from McUtils.GaussianInterface import GaussianLogReader
import numpy as np


class LogInterpreter:
    def __init__(self, *logs, moleculeObj=None, **kwargs):
        self.params = kwargs
        if len(logs) == 0:
            raise Exception('Nothing to interpret.')
        self.logs = logs
        self.molecule = moleculeObj
        self.method = self.molecule.method
        self.scancoord_1 = self.molecule.scanCoords[0]
        self.scancoord_2 = self.molecule.scanCoords[1]
        self._atomnum = None  # should be atomic numbers
        self._energies = None  # np.ndarray of electronic energies from Gaussian fillvalues to be square
        self._rawenergies = None  # np.ndarray of electronic energies ONLY FROM GAUSSIAN
        self._finite_energies = None  # np.ndarray of electronic energies from bottom of oo wells
        self._cartesians = None  # dictionary of cartesian coordinates keyed by (x, y) distances
        self._zmatrices = None   # dictionary of z-matrix values keyed by (x, y) distances
        self._dipoles = None     # dictionary of dipole values keyed by (x, y) distances

    @property
    def energies(self):
        if self._energies is None:
            self._energies = self.get_electronic_energy(**self.params)
        return self._energies

    @property
    def rawenergies(self):
        if self._rawenergies is None:
            self._rawenergies = self.get_electronic_energy(rawenergies=True, **self.params)
        return self._rawenergies

    @property
    def finite_energies(self):
        if self._finite_energies is None:
            self._finite_energies = self.get_finitedata()
        return self._finite_energies

    @property
    def cartesians(self):
        if self._cartesians is None:
            if len(self.logs) == 1:
                self._cartesians = self.get_scoords()
            else:
                self._cartesians = self.get_scoords(midpoint=True)
        return self._cartesians

    @property
    def atomnum(self):
        if self._atomnum is None:
            self._atomnum, self._zmatrices = self.get_zmats(**self.params)
        return self._atomnum

    @property
    def zmatrices(self):
        if self._zmatrices is None:
            self._atomnum, self._zmatrices = self.get_zmats()  # **self.params
        return self._zmatrices

    @property
    def dipoles(self):
        if self._dipoles is None:
            self._dipoles = self.get_dips(**self.params)
        return self._dipoles

    def get_electronic_energy(self, shift=True, optimized=None, rawenergies=False, **params):
        """Pulls Energies from the "Summary" portion of a log file. Shifts the potential energy to 0 to avoid
         problems with DVR later and extrapolates to a regular sized rudy grid to avoid problems with
         interpolation /sizing/ etc.
        :return: an array (col0: scancoord_1(ang), col1: scancoord_2(ang), col2: energy(shifted hartrees))
        :rtype: np.ndarray """
        from McUtils.Zachary.Interpolator import Interpolator
        ens = []
        crds = []
        full_grid = np.array(list(self.cartesians.keys()))
        if optimized is None:
            raise Exception("No energy type specified.")
        for log in self.logs:
            if optimized:
                with GaussianLogReader(log) as reader:
                    parse = reader.parse("OptimizedScanEnergies")
                energy_array, coords = parse["OptimizedScanEnergies"]
                roo = coords['Roo12']
                roh = coords['MrOH']
                if 'far' in log:
                    roh = -roh
                crds.append(np.column_stack((roo, roh)))
                ens.append(energy_array)
            else:
                with GaussianLogReader(log) as reader:
                    parse = reader.parse("ScanEnergies")
                vals = parse["ScanEnergies"]["values"][:, 3]  # N MrOH SCF MP2
                ens.append(vals)
        energy = np.concatenate(ens)
        if len(crds) > 0:
            coord_array = np.concatenate(crds)
            idx = np.lexsort((coord_array[:, 0], coord_array[:, 1]))
            energy = energy[idx]
            idx = np.lexsort((full_grid[:, 0], full_grid[:, 1]))
            full_grid = full_grid[idx]

        energyF = np.column_stack((full_grid, energy))
        idx = np.lexsort((full_grid[:, 1], full_grid[:, 0]))
        out = energyF[idx]
        if shift:
            out[:, 2:] -= min(out[:, 2:])  # shift gaussian numbers to zero HERE to avoid headaches later.
        else:
            pass
        if rawenergies:
            energyArray = out
        else:
            sq_grid, sq_vals = Interpolator(out[:, :2], out[:, 2], interpolation_function=lambda: "ignore").\
                regular_grid(interp_kind='cubic', fillvalues=True)
            energyArray = np.column_stack((sq_grid, sq_vals))
        return energyArray

    def get_electronic_energyOH(self, shift=True, optimized=None, rawenergies=False, **params):
        """USE IF WE EVER NEED OH SCANS BACK
         Pulls Energies from the "Summary" portion of a log file. Shifts the potential energy to 0 to avoid
         problems with DVR later and extrapolates to a regular sized rudy grid to avoid problems with
         interpolation /sizing/ etc.
        :return: an array (col0: scancoord_1(ang), col1: scancoord_2(ang), col2: energy(shifted hartrees))
        :rtype: np.ndarray """
        from McUtils.Zachary.Interpolator import Interpolator
        ens = []
        if optimized is None:
            raise Exception("No energy type specified.")
        for log in self.logs:
            if optimized:
                with GaussianLogReader(log) as reader:
                    parse = reader.parse("OptimizedScanEnergies")
                energy_array, coords = parse["OptimizedScanEnergies"]
                roo = coords['Roo12']
                roh = coords['Roh']
                ens.append(np.column_stack((roo, roh, energy_array)))
            else:
                with GaussianLogReader(log) as reader:
                    parse = reader.parse("ScanEnergies")
                just_the_val = parse["ScanEnergies"]["values"][:, 1:]  # returns only the MP2 energy
                just_the_val = np.concatenate((just_the_val[:, :2], just_the_val[:, [-1]]), axis=1)
                ens.append(just_the_val)
            energy = np.concatenate(ens)
            energy[:, 0] *= 2
            idx = np.lexsort((energy[:, 1], energy[:, 0]))
            energyF = energy[idx]
            row_mask = np.append([True], np.any(np.diff(energyF, axis=0), 1))
            out = energyF[row_mask]
        if shift:
            out[:, 2:] -= min(out[:, 2:])  # shift gaussian numbers to zero HERE to avoid headaches later.
        else:
            pass
        if rawenergies:
            energyArray = out
        else:
            sq_grid, sq_vals = Interpolator(out[:, :2], out[:, 2], interpolation_function=lambda: "ignore").\
                regular_grid(interp_kind='cubic', fillvalues=True)
            energyArray = np.column_stack((sq_grid, sq_vals))
        return energyArray

    def get_finitedata(self):
        """Should be able to pull energies from FD Scan log files and create a dictionary..
        implementation should plot to check"""
        import os
        import glob as glob
        FD_dir = os.path.join(self.molecule.mol_dir, 'Finite Scan Data', "oneD")
        FD_scans = sorted(glob.glob(os.path.join(FD_dir, ('Egraph_%s_*.dat' % self.method))))
        # oos = np.unique(np.array(list(self.cartesians.keys()))[:, 0])
        oos = np.arange(1.9696, 4.1296, 0.12)
        finite_energies = np.empty((0, 3), int)
        for oo, dat in zip(oos, FD_scans):
            pts = np.loadtxt(dat)
            oh = pts[:, 0]
            energy = pts[:, 1]
            rep_oo = np.repeat(oo, len(oh))
            vals = np.column_stack((rep_oo, oh, energy))
            finite_energies = np.append(finite_energies, vals, axis=0)
        finite_energies[:, 2:] -= min(finite_energies[:, 2:])
        return finite_energies

    def get_scoords(self, midpoint=False):
        """ pulls the optimized (if applicable) structures from  a 2D Gaussian scan.
        :return: atomnum:
        :return: cartesian coordinates at STANDARD optimized geometry keyed by the (scancoord_1, scancoord_2) distances.
        :rtype: OrderedDict
        """
        from MolecularSys import MolecularOperations
        from collections import OrderedDict
        struct = OrderedDict()
        for log in self.logs:
            with GaussianLogReader(log) as reader:
                parse = reader.parse("StandardCartesianCoordinates")
            atomnum, ccs = parse["StandardCartesianCoordinates"]
            xdists = MolecularOperations.calculateBonds(ccs, *self.scancoord_1)
            xdists = np.around(xdists, 4)

            ydists = MolecularOperations.calculateBonds(ccs, *self.scancoord_2)
            ydists = np.around(ydists, 4)
            if midpoint:
                if "flipped" in log:
                    MrOH = -1*((xdists / 2) - ydists)
                    MrOH = np.around(MrOH, 4)
                else:
                    MrOH = (xdists/2) - ydists
                    MrOH = np.around(MrOH, 4)
                coords = zip(xdists, MrOH, ccs)
            else:
                coords = zip(xdists, ydists, ccs)
            struct.update((((x, y), cc) for x, y, cc in coords))
        return struct

    def get_zmats(self, zmat_coord=None):
        """ pulls the Z-Matrices from a 1D Gaussian scan
        :param zmat_coord: x-coordinate of the scan
        :type zmat_coord: tuple
        :return: Z-matrix values keyed by the coordinate distances.
        :rtype: OrderedDict
        """
        from collections import OrderedDict
        struct = OrderedDict()
        for log in self.logs:
            with GaussianLogReader(log) as reader:
                parse = reader.parse("ZMatrices")
            zmcs = parse["ZMatrices"]
            atom_num = zmcs[1]
            vals = zmcs[2]
            xps = vals[:, 0, zmat_coord]
            xdists = np.around(xps, 4)

            things = zip(xdists, vals)
            struct.update(((oo, zm) for oo, zm in things))
        return atom_num, struct

    def get_crds_for_one(self, use_zmat=False, xdist=None, ydist=None):
        # should work with either cartesian coordinates or z matrices, does it?? who knows.
        """Takes the entire dictionary of structures of parses it for only one oo distance.
            :param use_zmat: if True, pulls Z-matrix value instead of Cartesian Coordinates
            :type  use_zmat: False
            :param xdist: x distance of coordinates pulled (ang)
            :type xdist: float (4 decimal places)
            :param ydist: y distance of coordinates pulled (ang)
            :type ydist: float (4 decimal places)
            :returns crds_for_one: standard cartesian coordinates for structure with xdist, ydist"""
        crds_for_one = ()
        if use_zmat:
            pos = list(self.zmatrices.keys())
            for x in pos:
                if x[0] == xdist and x[1] == ydist:
                    crds_for_one = self.zmatrices[x]
        else:
            pos = list(self.cartesians.keys())
            for x in pos:
                if x[0] == xdist and x[1] == ydist:
                    crds_for_one = self.cartesians[x]
        return crds_for_one

    def get_dips(self, optimized=False, **params):
        """Pulls the dipoles from a list of Gaussian log files
        :param optimized: if True, the list of log files are optimization scans
        :type optimized: bool
        :return: Dipole values keyed by the coordinate distances.
        :rtype: OrderedDict
        """
        from collections import OrderedDict
        struct = OrderedDict()
        for log in self.logs:
            with GaussianLogReader(log) as reader:
                parse = reader.parse(("OptimizedDipoleMoments", "DipoleMoments"))
            if optimized:
                dips = parse["OptimizedDipoleMoments"]
                dips = np.array(list(dips))
                Findips = dips / 0.393456  # convert dipoles from au to debye IMMEDIATELY!
            else:
                dips = parse["DipoleMoments"]
                Findips = np.array(list(dips))
            coord = LogInterpreter(log, moleculeObj=self.molecule).cartesians.keys()
            struct.update(((c, d) for c, d in zip(coord, Findips)))
        return struct

    def cut_dictionary(self, midpoint=False):
        """ creates a dictionary of cuts from POTENTIAL ENERGY SURFACES (uses those coords and energies)
        :return: dictionary: {scancoord_1: (scancoord_2, energy)} (input units)
        :rtype: OrderedDict
        """
        from collections import OrderedDict
        pes = self.energies
        xvals = np.unique(self.energies[:, 0])
        slices = [pes[self.energies[:, 0] == xv] for xv in xvals]
        slices = [slICE[slICE[:, 1].argsort()] for slICE in slices]
        cuts = OrderedDict()
        for slICE in slices:
            if midpoint:
                slICE[:, 1] = (slICE[0, 0] / 2) - slICE[:, 1]
                slICE[:, 1] *= -1
            cuts[slICE[0, 0]] = slICE[:, 1:]
        return cuts

    def finite_dict(self, midpoint=False):
        from collections import OrderedDict
        pes = self.finite_energies
        xvals = np.unique(self.finite_energies[:, 0])
        slices = [pes[self.finite_energies[:, 0] == xv] for xv in xvals]
        slices = [slICE[slICE[:, 1].argsort()] for slICE in slices]
        cuts = OrderedDict()
        for slICE in slices:
            if midpoint:
                slICE[:, 1] = (slICE[0, 0] / 2) - slICE[:, 1]
                slICE[:, 1] *= -1
            cuts[slICE[0, 0]] = slICE[:, 1:]
        return cuts

    def minimum_pot(self):
        """Pulls the Scan Energies of a 1D Scan and shifts so that minimum is 0. So far hard coded to "Roo12" and "Roh"
        :return: array (col0: scancoord_1, col1: scancoord_2, col1: energy)
        :rtype: np.ndarray
        """
        import os
        scan_dir = os.path.join(self.molecule.mol_dir, '1D Scans')
        onedeescan = os.path.join(scan_dir, ("1D_%s_ooscan.log" % self.method))
        with GaussianLogReader(onedeescan) as reader:
            parse = reader.parse("OptimizedScanEnergies")
        energy_array, coords = (parse["OptimizedScanEnergies"])
        roo = coords['Roo12']*2
        roh = coords['Roh']
        ens = np.column_stack((roo, roh, energy_array))
        ens[:, 2:] -= min(ens[:, 2:])  # shift gaussian numbers to zero HERE to avoid headaches later.
        return ens[:18, :]

    def write_finitescan(self, config=None, variables=None):
        """ Writes and save a gaussian gjf file to conduct a scan with a small enough range for finite differencing
        :param config: use to change number of processors or memory allocated to job
        :type config: dict
        :param variables: set system constants
        :type variables: list of tuples
        :return: a gjf file
        :rtype: output file
        """
        from McUtils.GaussianInterface import GaussianJob
        xdists = np.array(list(self.zmatrices.keys()))
        zmcs = np.array(list(self.zmatrices.values()))
        default_config = dict(NProc=4, Mem='1000MB')
        default_config.update(config if config is not None else dict())
        yvals = zmcs[:, 3, 0]  # CHANGE THIS SOMEHOW
        for i in range(0, len(xdists)):
            # default_variables = [("dist4", (yvals[i] - 0.01), 4, .005)]
            # default_variables.extend(variables if variables is not None else [])
            GaussianJob(
                "Finite Difference scan jobs",
                description="FD Scan for Roo %.3f" % (xdists[i]*2),
                file='%s_finiteSPE_%.3f.gjf' % (self.method, xdists[i]*2),
                template="TemplateTerse.gjf",
                config=GaussianJob.Config(**default_config),
                job=GaussianJob.Job('Scan'),
                system=GaussianJob.System(
                    charge=1,
                    molecule=[self.method.atom_str, (self.atomnum, zmcs[i])],
                    vars=[GaussianJob.System.Variable("dist4", (yvals[i] - 0.01), 4, .005),
                          GaussianJob.System.Constant("angle4", 0.00001)]
                )).write()

