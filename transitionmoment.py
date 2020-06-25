import numpy as np


class TransitionMoment:
    def __init__(self, moleculeObj=None, dimension=None, OHDVRnpz=None, OODVRnpz=None, TwoDnpz=None, delta=False, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.method = self.molecule.method
        self.scanCoords = self.molecule.scanCoords
        self.logData = moleculeObj.logData
        self.delta = delta
        self._embeddedCoords = None
        self._embeddedDips = None
        self.TwoDres = np.load(TwoDnpz)
        self._TwoDDips = None
        self._TwoD_dms = None
        self.twoDexpanTypes = ["dipSurf", "cubic", "quad", "quadOH", "quadbilin", "lin", "linOH"]
        if dimension == "1D":
            self.dimension = "1D"
            self.OHDVRres = np.load(OHDVRnpz)
            self.OODVRres = np.load(OODVRnpz)
            self._mus = None
            self._tdms = None
        elif dimension == "2D":
            self.dimension = "2D"
        else:
            raise Exception("No TM dimensionality specified")

    @property
    def embeddedCoords(self):
        import os
        npcoordname = os.path.join(self.molecule.mol_dir, "structures", f"{self.method}_2D_embeddedCoords.npy")
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
        npfilename = os.path.join(self.molecule.mol_dir, "structures", f"{self.method}_2D_embeddedDipoles.npy")
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
    def TwoD_dms(self):
        if self._TwoD_dms is None:
            self._TwoD_dms = self.calc_all2Dmus()
        return self._TwoD_dms

    def embed(self):
        from MolecularSys import MolecularOperations
        newcoords = MolecularOperations(self.molecule).embeddedCoords
        newdips = MolecularOperations(self.molecule).embeddedDips
        return newcoords, newdips

    def savestructs(self):
        import os
        npfilename = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedDipoles.npy")
        npcoordname = os.path.join(self.molecule.mol_dir, "DVR Results", f"{self.method}_embeddedCoords.npy")
        np.save(npfilename, self.embeddedDips)
        np.save(npcoordname, self.embeddedCoords)

    def interp_2D_dipoles(self):
        from functools import reduce
        from operator import mul
        from Converter import Constants
        from MolecularSys import MolecularOperations
        from scipy.interpolate import griddata
        # in bohr
        oos = MolecularOperations(self.molecule).calculateBonds(self.embeddedCoords, *self.scanCoords[0])
        ohs = MolecularOperations(self.molecule).calculateBonds(self.embeddedCoords, *self.scanCoords[1])
        MrOH = (oos / 2) - ohs
        MrOH = np.around(MrOH, 4)
        dip_vecs = self.embeddedDips
        if self.dimension == "2D":
            bigGrid = Constants.convert(self.TwoDres["grid"][0], "angstroms", to_AU=True)
            npts = reduce(mul, bigGrid.shape[:-1], 1)
            grid = np.reshape(bigGrid, (npts, bigGrid.shape[-1]))
            new_dips = np.zeros((npts, 3))
            for j in np.arange(3):
                if self.delta:
                    Doos = oos - Constants.convert(self.molecule.OOmin, "angstroms", to_AU=True)
                    Dxhs = MrOH - Constants.convert(self.molecule.XHmin, "angstroms", to_AU=True)
                    new_dips[:, j] = griddata(np.column_stack((Doos, Dxhs)), dip_vecs[:, j], grid,
                                              method="cubic", fill_value=np.min(dip_vecs[:, j]))
                else:
                    new_dips[:, j] = griddata(np.column_stack((oos, MrOH)), dip_vecs[:, j], grid,
                                              method="cubic", fill_value=np.min(dip_vecs[:, j]))
        else:  # this should interpolate grid to 1D DVR # of MrOHs x PES # of OOs (500 x 35)
            potz = self.OHDVRres["potential"]
            rxh = Constants.convert(np.unique(potz[:, :, 0]), "angstroms", to_AU=True)
            cut_dict = self.logData.cut_dictionary()
            roo = Constants.convert(np.array(list(cut_dict.keys())), "angstroms", to_AU=True)
            new_grid = np.meshgrid(roo, rxh, indexing="ij")
            grid = np.column_stack((new_grid[0].flatten(), new_grid[1].flatten()))
            new_dips = np.zeros((len(grid), 3))
            for j in np.arange(3):
                new_dips[:, j] = griddata(np.column_stack((oos, MrOH)), dip_vecs[:, j], grid,
                                          method="cubic", fill_value=np.min(dip_vecs[:, j]))
        return grid, new_dips  # bohr & debye

    def calc_all2Dmus(self):
        from TDMexpansions import TM2Dexpansion
        Grid = self.TwoDDips[0]  # oo/xh bohr
        oos = len(np.unique(Grid[:, 0]))
        xhs = len(np.unique(Grid[:, 1]))
        Dips = self.TwoDDips[1]
        if self.dimension == "2D":
            Pot = self.TwoDres["potential"]  # oo/xh
        else:
            potz = self.OHDVRres["potential"]
            Pot = potz[:, :, 1]
        square = np.reshape(Grid, (oos, xhs, 2))
        squarepot = np.reshape(Pot, (oos, xhs))
        squaredips = np.reshape(Dips, (oos, xhs, 3))
        mini = np.unravel_index(np.argmin(squarepot, axis=None), squarepot.shape)
        pickrangex = np.arange(mini[0] - 2, mini[0] + 3)
        pickrangey = np.arange(mini[1] - 2, mini[1] + 3)
        FDvalues = np.zeros((3, 5, 5))
        for j in np.arange(3):
            for i, v in enumerate(pickrangex):
                FDvalues[j, i, :] = squaredips[v, pickrangey, j]

        params = dict()
        # params["eqDipole"] = np.array((FDvalues[0, 2, 2], FDvalues[1, 2, 2], FDvalues[2, 2, 2]))
        # params["eqDipole"] = np.array((0.42059027, 1.597753, -0.01960174))  # place in EQ Dipole from the small scan
        params["eqDipole"] = np.array((0.60520204, 0.96477493, 1.51323868))  # EQ Dipole from the small scan (tri)
        # print(params["eqDipole"])
        fd_ohs = square[pickrangex[2], pickrangey, 1]
        fd_oos = square[pickrangex, pickrangey[2], 0]
        if self.delta:
            params["delta_roh"] = Grid[:, 1]
            params["delta_Roo"] = Grid[:, 0]
        else:
            params["delta_roh"] = Grid[:, 1] - fd_ohs[2]
            params["delta_Roo"] = Grid[:, 0] - fd_oos[2]

        # xderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[0])
        # yderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[1])
        # zderivs = self.calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues[2])
        # derivs = {'x': xderivs, 'y': yderivs, 'z': zderivs}
        # np.savez("DipCoefstest.npz", x=xderivs, y=yderivs, z=zderivs)

        derivs = np.load("DipCoefsH9O4pls_smallscan.npz", allow_pickle=True)
        newDerivs = {k: derivs[k].item() for k in ["x", "y", "z"]}
        # newDerivs = {k:{l:1 for l in newDerivs[k]} for k in newDerivs}
        # print("x-derivs:", newDerivs["x"])
        # print("y-derivs:", newDerivs["y"])
        # print("z-derivs:", newDerivs["z"])
        twodeetdms = dict()
        twodeetdms["dipSurf"] = Dips
        twodeetdms["cubic"] = TM2Dexpansion.cubic_DM(params, newDerivs)
        twodeetdms["quad"] = TM2Dexpansion.quad_DM(params, newDerivs)
        twodeetdms["quadOH"] = TM2Dexpansion.quadOH_DM(params, newDerivs)
        twodeetdms["quadbilin"] = TM2Dexpansion.quadBILIN_DM(params, newDerivs)
        twodeetdms["lin"] = TM2Dexpansion.lin_DM(params, newDerivs)
        twodeetdms["linOH"] = TM2Dexpansion.linOH_DM(params, newDerivs)
        return Grid, twodeetdms

    def make1D_DipStruct(self):
        grid, twodee_DMs = self.TwoD_dms
        # cut_dict = self.logData.cut_dictionary()
        # roos = np.array(list(cut_dict.keys()))
        # idx = np.argwhere([np.around(Grid[:, 0], 2) == np.around(r, 2) for r in roos])
        # grid = Grid[idx[:, 1]]
        dip_expand_cuts = dict()
        for i, t in enumerate(self.twoDexpanTypes):
            dips = twodee_DMs[t]
            # dips = dips[idx[:, 1]]
            res = np.column_stack((grid[:, 0], grid[:, 1], dips[:, 0], dips[:, 1], dips[:, 2]))
            res = res[res[:, 0].argsort()]
            cuts = res.reshape((len(np.unique(grid[:, 0])), len(np.unique(grid[:, 1])), 5))
            cuts = np.array([cut[cut[:, 1].argsort()] for cut in cuts])
            dip_expand_cuts[t] = cuts
        return dip_expand_cuts

    def psi_trans(self):
        """calculates the transition moment at each OO value. Returns the 2d grid (nOO, nOH, 2) and the TM at each OO"""
        interp_dict = self.make1D_DipStruct()
        ohWfns = self.OHDVRres["wfns_array"]
        grids = dict()
        muses = dict()
        for i, t in enumerate(self.twoDexpanTypes):
            val = interp_dict[t]
            grids[t] = val[:, :, :2]
            dip_vecs = val[:, :, 2:]
            mus = np.zeros((len(dip_vecs), 3))
            for k in np.arange(len(dip_vecs)):  # loop through cuts
                for j in np.arange(3):  # loop through x, y, z
                    gs_wfn = ohWfns[k, :, 0].T
                    es_wfn = ohWfns[k, :, 1].T
                    es_wfn_t = es_wfn.reshape(-1, 1)
                    soup = np.diag(dip_vecs[k, :, j]).dot(es_wfn_t)
                    mu = gs_wfn.dot(soup)
                    mus[k, j] = mu
            muses[t] = mus
        return grids, muses

    def calc_all1Dmus(self, npts=7):
        from Converter import Constants
        grids = self.mus[0]
        x = grids["dipSurf"][:, 0, 0]
        muses = self.mus[1]
        bigPot = self.OODVRres["potential"][0]
        bigGrid = Constants.convert(bigPot[:, 0], "angstroms", to_AU=True)
        mini = np.argmin(bigPot[:, 1])
        xmin = np.abs(x - bigGrid[mini]).argmin()
        minix = x[xmin - (npts // 2):xmin + (npts // 2) + 1]
        xeqX = bigGrid - minix[npts//2]
        all_mus = dict()
        for i, t in enumerate(self.twoDexpanTypes):
            mus = muses[t]
            minimus = mus[xmin - (npts // 2):xmin + (npts // 2) + 1]
            poly_mus = np.zeros((len(xeqX), 3))
            for j in np.arange(3):  # loop through components
                poly_vals = np.polyfit(minix - minix[npts // 2], minimus[:, j].T, npts - 1)
                p = np.poly1d(poly_vals)
                poly_mus[:, j] = p(xeqX)
            all_mus[t] = poly_mus
        return bigGrid, all_mus

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
