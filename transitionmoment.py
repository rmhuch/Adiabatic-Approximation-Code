import numpy as np


class TransitionMoment:
    def __init__(self, moleculeObj=None, dimension=None, OHDVRnpz=None, OODVRnpz=None, TwoDnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.method = self.molecule.method
        self.scanCoords = self.molecule.scanCoords
        self._logData = None
        self._embeddedCoords = None
        self._embeddedDips = None
        if dimension == "1D":
            self.OHDVRres = np.load(OHDVRnpz)
            self.OODVRres = np.load(OODVRnpz)
            self._mus = None
        elif dimension == "2D":
            self.TwoDres = np.load(TwoDnpz)
            self._TwoDDips = None
        else:
            raise Exception("No TM dimensionality specified")

    @property
    def logData(self):
        if self._logData is None:
            from GaussianHandler import LogInterpreter
            if self.method == "rigid":
                optBool = False
            else:
                optBool = True
            self._logData = LogInterpreter(*self.molecule.scanLogs, moleculeObj=self.molecule, optimized=optBool)
        return self._logData

    @property
    def embeddedCoords(self):
        import os
        npcoordname = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedCoords.npy")
        if self._embeddedCoords is None:
            if os.path.exists(npcoordname):
                self._embeddedCoords = np.load(npcoordname)
            else:
                self._embeddedCoords, self._embeddedDips = self.embed()
                self.savestructs()
        return self._embeddedCoords

    @property
    def embeddedDips(self):
        import os
        npfilename = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedDipoles.npy")
        if self._embeddedDips is None:
            if os.path.exists(npfilename):
                self._embeddedDips = np.load(npfilename)
            else:
                self._embeddedCoords, self._embeddedDips = self.embed()
                self.savestructs()
        return self._embeddedDips

    @property
    def mus(self):
        if self._mus is None:
            self._mus = self.run_tm()
        return self._mus

    @property
    def TwoDDips(self):
        if self._TwoDDips is not None:
            self._TwoDDips = self.interp_2D_dipoles()
        return self._TwoDDips

    def embed(self):
        from MolecularSys import MolecularOperations
        newcoords = MolecularOperations(self.molecule).embeddedCoords
        newdips = MolecularOperations(self.molecule).embeddedDips
        return newcoords, newdips

    def run_tm(self):
        """returns {mus} uninterpolated transition moments and {interp_mus} interpolated TMs"""
        from scipy import interpolate
        mus = self.psi_trans()
        ooWfns = self.OODVRres["wfns_array"]
        oopot = self.OODVRres["potential"]
        wfnAmpIdx = np.argwhere(ooWfns[0, :, 0] > 1E-5)
        ooWfnAmp = oopot[0, wfnAmpIdx, 0]
        interp_mus = np.zeros((len(ooWfnAmp), 4))
        interp_mus[:, 0] = ooWfnAmp.T
        for i in np.arange(3):
            # tck = interpolate.splrep(mus[:, 0], mus[:, i+1], s=0)
            # interp_mus[:, i+1] = interpolate.splev(ooWfnAmp.T, tck, der=0)
            f = interpolate.interp1d(mus[2:, 0], mus[2:, i+1], kind="cubic", fill_value="extrapolate", bounds_error=False)
            interp_mus[:, i + 1] = f(ooWfnAmp.T)
        return mus, interp_mus
    
    def makeDipStruct(self, preEmbed=False):
        from McUtils.Zachary.Interpolator import Interpolator
        ignore_func = lambda: "ignore"
        scangrid = np.array(list(self.logData.dipoles.keys()))
        if preEmbed:
            dip_vecs = np.array(list(self.logData.dipoles.values()))
            grid, xvals = Interpolator(scangrid, dip_vecs[:, 0],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
            grid, yvals = Interpolator(scangrid, dip_vecs[:, 1],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
            grid, zvals = Interpolator(scangrid, dip_vecs[:, 2],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
        else:
            grid, xvals = Interpolator(scangrid, self.embeddedDips[:, 0],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
            grid, yvals = Interpolator(scangrid, self.embeddedDips[:, 1],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
            grid, zvals = Interpolator(scangrid, self.embeddedDips[:, 2],
                                       interpolation_function=ignore_func).regular_grid(fillvalues=True)
        res = np.column_stack((grid[:, 0], grid[:, 1], xvals, yvals, zvals))
        res = res[res[:, 0].argsort()]
        cuts = res.reshape((len(np.unique(grid[:, 0])), len(np.unique(grid[:, 1])), 5))
        cuts = np.array([cut[cut[:, 1].argsort()] for cut in cuts])
        return cuts

    def savestructs(self):
        import os
        npfilename = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedDipoles.npy")
        npcoordname = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedCoords.npy")
        np.save(npfilename, self.embeddedDips)
        np.save(npcoordname, self.embeddedCoords)

    def interp_2D_dipoles(self):
        from PotentialHandlers import Potentials2D
        from functools import reduce
        from operator import mul
        val = self.makeDipStruct()
        rohs = val[0, :, 1]
        roos = val[:, 0, 0]
        dip_vecs = val[:, :, 2:]
        dip_vecs = np.reshape(dip_vecs, (dip_vecs.shape[0]*dip_vecs.shape[1], 3))
        bigGrid = self.TwoDres["grid"][0]
        npts = reduce(mul, bigGrid.shape[:-1], 1)
        grid = np.reshape(bigGrid, (npts, bigGrid.shape[-1]))
        new_dips = np.zeros((npts, 3))
        for j in np.arange(3):
            extrap_funct = Potentials2D().exterp2d(np.column_stack((roos, rohs)), dip_vecs[:, j])
            new_dips[:, j] = extrap_funct(grid)
            # from McUtils.Plots import ListContourPlot
            # ListContourPlot(np.column_stack((grid, new_dips[:, j]))).show()
        return new_dips

    def interp_dipoles(self):
        from scipy import interpolate
        val = self.makeDipStruct()
        rohs = val[0, :, 1]
        roos = val[:, 0, 0]
        dip_vecs = val[:, :, 2:]
        potz = self.OHDVRres["potential"]
        ohWfns = self.OHDVRres["wfns_array"]
        wfnAmpIdx = np.argwhere(ohWfns[4, :, 0] > 1E-5)
        ohWfnAmp = potz[4, wfnAmpIdx, 0]
        new_dip_vals = np.zeros((len(dip_vecs), len(ohWfnAmp), 5))
        for i, roo in enumerate(roos):
            for j in np.arange(3):
                dip_vals = dip_vecs[i, :, j]
                # mids = roo/2 - val[i, :, 1]
                # mids *= -1
                # tck = interpolate.splrep(rohs, dip_vals, s=0)
                f = interpolate.interp1d(rohs, dip_vals,
                                         kind="cubic", fill_value=(dip_vals[0], dip_vals[-1]), bounds_error=False)
                new_dip_vals[i, :, 0] = np.repeat(val[i, 0, 0], len(potz[i, wfnAmpIdx, 0]))
                new_dip_vals[i, :, 1] = potz[i, wfnAmpIdx, 0].T
                # new_dip_vals[i, :, j+2] = interpolate.splev(potz[i, wfnAmpIdx, 0].T, tck, der=0)
                new_dip_vals[i, :, j+2] = f(potz[i, wfnAmpIdx, 0].T)
        return new_dip_vals

    def psi_trans(self):
        val = self.interp_dipoles()
        dip_vecs = val[:, :, 2:]
        mus = np.zeros((len(dip_vecs), 4))
        ohWfns = self.OHDVRres["wfns_array"]
        wfnAmpIdx = np.argwhere(ohWfns[4, :, 0] > 1E-5)
        for k in np.arange(len(dip_vecs)):  # loop through cuts
            mus[k, 0] = val[k, 0, 0]
            for j in np.arange(3):  # loop through x, y, z
                gs_wfn = ohWfns[k, wfnAmpIdx, 0].T
                es_wfn = ohWfns[k, wfnAmpIdx, 1].T
                es_wfn_t = es_wfn.reshape(-1, 1)
                soup = np.diag(dip_vecs[k, :, j]).dot(es_wfn_t)
                mu = gs_wfn.dot(soup)
                mus[k, j+1] = mu
        # mus[:, 1:] *= -1
        return mus

