import numpy as np


class TransitionMoment:
    def __init__(self, moleculeObj=None, dimension=None, OHDVRnpz=None, OODVRnpz=None, TwoDnpz=None, min=False, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.method = self.molecule.method
        self.scanCoords = self.molecule.scanCoords
        self.logData = moleculeObj.logData
        self.min = min
        self._embeddedCoords = None
        self._embeddedDips = None
        if dimension == "1D":
            self.dimension = "1D"
            self.OHDVRres = np.load(OHDVRnpz)
            self.OODVRres = np.load(OODVRnpz)
            self._mus = None
            self._tdms = None
        elif dimension == "2D":
            self.dimension = "2D"
            self.TwoDres = np.load(TwoDnpz)
            self._TwoDDips = None
            self._TwoDtdms = None
        else:
            raise Exception("No TM dimensionality specified")

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
            self._mus = self.psi_trans()
        return self._mus

    @property
    def tdms(self):
        if self._tdms is None:
            self._tdms = self.calc_all1Dmus()
        return self._tdms

    @property
    def TwoDDips(self):
        if self._TwoDDips is None:
            self._TwoDDips = self.interp_2D_dipoles()
        return self._TwoDDips

    @property
    def TwoDtdms(self):
        if self._TwoDtdms is None:
            self._TwoDtdms = self.calc_all2Dmus()
        return self._TwoDtdms

    def embed(self):
        from MolecularSys import MolecularOperations
        newcoords = MolecularOperations(self.molecule).embeddedCoords
        newdips = MolecularOperations(self.molecule).embeddedDips
        return newcoords, newdips

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
        import os
        from functools import reduce
        from operator import mul
        from Converter import Constants
        from MolecularSys import MolecularOperations
        from scipy.interpolate import griddata

        npfilename = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_2D_embeddedDipoles.npy")
        npcoordname = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_2D_embeddedCoords.npy")
        if os.path.exists(npfilename):
            newdips = np.load(npfilename)
            newcoords = np.load(npcoordname)
        else:
            newcoords, newdips = MolecularOperations(self.molecule).many_rotations(**self.molecule.embed_dict)
            np.save(npfilename, newdips)
            np.save(npcoordname, newcoords)
        oos = Constants.convert(MolecularOperations(self.molecule).calculateBonds(newcoords, *self.scanCoords[0]),
                                "angstroms", to_AU=False)
        ohs = Constants.convert(MolecularOperations(self.molecule).calculateBonds(newcoords, *self.scanCoords[1]),
                                "angstroms", to_AU=False)
        MrOH = (oos / 2) - ohs
        MrOH = np.around(MrOH, 4)
        dip_vecs = newdips
        bigGrid = self.TwoDres["grid"][0]
        npts = reduce(mul, bigGrid.shape[:-1], 1)
        grid = np.reshape(bigGrid, (npts, bigGrid.shape[-1]))
        new_dips = np.zeros((npts, 3))
        for j in np.arange(3):
            if self.min:
                Doos = oos - self.molecule.OOmin
                Dxhs = MrOH - self.molecule.XHmin
                new_dips[:, j] = griddata(np.column_stack((Doos, Dxhs)), dip_vecs[:, j], grid,
                                          method="cubic", fill_value=np.min(dip_vecs[:, j]))
            else:
                new_dips[:, j] = griddata(np.column_stack((oos, MrOH)), dip_vecs[:, j], grid,
                                          method="cubic", fill_value=np.min(dip_vecs[:, j]))
        return grid, new_dips

    def interp_dipoles(self):
        from scipy import interpolate
        val = self.makeDipStruct()
        rohs = val[0, :, 1]
        roos = val[:, 0, 0]
        dip_vecs = val[:, :, 2:]
        potz = self.OHDVRres["potential"]
        # roh potential (18, 500, 2)
        ohWfns = self.OHDVRres["wfns_array"]
        new_dip_vals = np.zeros((len(dip_vecs), ohWfns.shape[1], 5))
        # roo roh x y z  (18, 500, 5)
        for i, roo in enumerate(roos):
            for j in np.arange(3):
                dip_vals = dip_vecs[i, :, j]
                f = interpolate.interp1d(rohs, dip_vals, kind="cubic", fill_value=(dip_vals[0], dip_vals[-1]),
                                         bounds_error=False)
                new_dip_vals[i, :, 0] = np.repeat(val[i, 0, 0], new_dip_vals.shape[1])
                new_dip_vals[i, :, 1] = potz[i, :, 0].T
                new_dip_vals[i, :, j + 2] = f(potz[i, :, 0].T)
        return new_dip_vals

    def psi_trans(self):
        """calculates the transition moment at each OO value. Returns the 2d grid (nOO, nOH, 2) and the TM at each OO"""
        val = self.interp_dipoles()
        grid = val[:, :, :2]
        dip_vecs = val[:, :, 2:]
        mus = np.zeros((len(dip_vecs), 3))
        ohWfns = self.OHDVRres["wfns_array"]
        for k in np.arange(len(dip_vecs)):  # loop through cuts
            for j in np.arange(3):  # loop through x, y, z
                gs_wfn = ohWfns[k, :, 0].T
                es_wfn = ohWfns[k, :, 1].T
                es_wfn_t = es_wfn.reshape(-1, 1)
                soup = np.diag(dip_vecs[k, :, j]).dot(es_wfn_t)
                mu = gs_wfn.dot(soup)
                mus[k, j] = mu
        return grid, mus

    def calc_all1Dmus(self, npts=7):
        from TDMexpansions import TM1Dexpansion
        x = self.mus[0][:, 0, 0]
        mus = self.mus[1]
        bigPot = self.OODVRres["potential"][0]
        bigGrid = bigPot[:, 0]
        mini = np.argmin(bigPot[:, 1])
        xmin = np.abs(x - bigGrid[mini]).argmin()
        minix = x[xmin - (npts // 2):xmin + (npts // 2) + 1]
        minimus = mus[xmin - (npts // 2):xmin + (npts // 2) + 1]

        xeqX = bigGrid - bigPot[mini, 0]
        poly_vals = np.zeros((3, npts))
        for i in np.arange(3):  # loop through components
            poly_vals[i, :] = np.polyfit(minix - minix[npts // 2], minimus[:, i].T, npts - 1)

        tdms = dict()
        tdms["poly"] = TM1Dexpansion.polyTDM(xeqX, poly_vals)
        tdms["cubic"] = TM1Dexpansion.cubicTDM(xeqX, poly_vals)
        tdms["quad"] = TM1Dexpansion.quadTDM(xeqX, poly_vals)
        tdms["lin"] = TM1Dexpansion.linTDM(xeqX, poly_vals)
        tdms["const"] = TM1Dexpansion.constTDM(xeqX, poly_vals)
        return bigGrid, tdms

    def calc_all2Dmus(self):
        from TDMexpansions import TM2Dexpansion
        Grid = self.TwoDDips[0]
        Dips = self.TwoDDips[1]
        Pot = self.TwoDres["potential"]
        oos = len(np.unique(Grid[:, 0]))
        ohs = len(np.unique(Grid[:, 1]))
        square = np.reshape(Grid, (oos, ohs, 2))
        squarepot = np.reshape(Pot, (oos, ohs))
        squaredips = np.reshape(Dips, (oos, ohs, 3))
        mini = np.unravel_index(np.argmin(squarepot, axis=None), squarepot.shape)
        pickrangex = np.arange(mini[0] - 2, mini[0] + 3)
        pickrangey = np.arange(mini[1] - 2, mini[1] + 3)
        FDvalues = np.zeros((3, 5, 5))
        for j in np.arange(3):
            for i, v in enumerate(pickrangex):
                FDvalues[j, i, :] = squaredips[v, pickrangey, j]

        params = dict()
        params["eqDipole"] = np.array((FDvalues[0, 2, 2], FDvalues[1, 2, 2], FDvalues[2, 2, 2]))
        fd_ohs = square[pickrangex[2], pickrangey, 1]
        fd_oos = square[pickrangex, pickrangey[2], 0]
        FDgrid = np.array(np.meshgrid(fd_oos, fd_ohs)).T
        params["delta_roh"] = Grid[:, 1] - fd_ohs[2]
        params["delta_Roo"] = Grid[:, 0] - fd_oos[2]

        xderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[0])
        yderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[1])
        zderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[2])
        derivs = {'x': xderivs, 'y': yderivs, 'z': zderivs}

        twodeetdms = dict()
        twodeetdms["poly"] = Dips
        twodeetdms["cubic"] = TM2Dexpansion.cubicTDM(params, derivs)
        twodeetdms["quad"] = TM2Dexpansion.quadTDM(params, derivs)
        twodeetdms["quadOH"] = TM2Dexpansion.quadOHtdm(params, derivs)
        twodeetdms["quadbilin"] = TM2Dexpansion.quadBILINtdm(params, derivs)
        twodeetdms["lin"] = TM2Dexpansion.linTDM(params, derivs)
        twodeetdms["const"] = TM2Dexpansion.constTDM(params, derivs)
        return Grid, twodeetdms

    @staticmethod
    def calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues):
        from McUtils.Zachary import finite_difference
        derivs = dict()
        derivs["firstOH"] = finite_difference(fd_ohs, FDvalues[2, :], 1,
                                              end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["firstOO"] = finite_difference(fd_oos, FDvalues[:, 2], 1,
                                              end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["secondOH"] = finite_difference(fd_ohs, FDvalues[2, :], 2,
                                               end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["secondOO"] = finite_difference(fd_oos, FDvalues[:, 2], 2,
                                               end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["thirdOH"] = finite_difference(fd_ohs, FDvalues[2, :], 3,
                                              end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["thirdOO"] = finite_difference(fd_oos, FDvalues[:, 2], 3,
                                              end_point_precision=0, stencil=5, only_center=True)[0]
        derivs["mixedOHOO"] = finite_difference(FDgrid, FDvalues, (1, 1), stencil=(5, 5),
                                                accuracy=0, end_point_precision=0, only_center=True)[0, 0]
        derivs["mixedOHOOOO"] = finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5),
                                                  accuracy=0, end_point_precision=0, only_center=True)[0, 0]
        derivs["mixedOHOHOO"] = finite_difference(FDgrid, FDvalues, (2, 1), stencil=(5, 5),
                                                  accuracy=0, end_point_precision=0, only_center=True)[0, 0]
        return derivs
