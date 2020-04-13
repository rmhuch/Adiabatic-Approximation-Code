import numpy as np

class CubicHarmonic:
    def __init__(self, moleculeObj, omegaOO, omegaOH, FancyF=None):
        self.molecule = moleculeObj
        self.omegaOO = omegaOO
        self.omegaOH = omegaOH
        self.FancyF = FancyF

    def find_FancyF(self):
        import os
        from Converter import Constants
        from McUtils.Zachary import finite_difference
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        freqoo = Constants.convert(self.omegaOO, "wavenumbers", to_AU=True)
        muOO = mO / 2
        freqoh = Constants.convert(self.omegaOH, "wavenumbers", to_AU=True)
        muOH = ((2 * mO) * mH) / ((2 * mO) + mH)
        fs_dir = os.path.join(self.molecule.mol_dir, "Finite Scan Data")
        if self.molecule.MoleculeName == "H9O4pls":
            finite_vals = np.loadtxt(f"{fs_dir}/2D_finiteSPEtet_01_008.dat", skiprows=7)
        elif self.molecule.MoleculeName == "H7O3pls":
            finite_vals = np.loadtxt(f"{fs_dir}/2D_finiteSPEtri_01_008.dat", skiprows=7)
        else:
            raise Exception("Can't compute FancyF of that molecule")
        finite_vals = finite_vals[:, 2:]
        finite_vals[:, 0] *= 2  # multiply OO/2 by 2 for OO values
        finite_vals[:, :2] = Constants.convert(finite_vals[:, :2], "angstroms", to_AU=True)  # convert OO/OH to bohr
        finite_vals[:, 2] -= min(finite_vals[:, 2])  # shift minimum to 0 so that energies match
        FDgrid = np.array(
            np.meshgrid(np.unique(finite_vals[:, 0]), np.unique(finite_vals[:, 1]))).T  # create mesh OO/OH
        FDvalues = np.reshape(finite_vals[:, 2], (5, 5))
        FrrR = finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5), accuracy=0, end_point_precision=0,
                                 only_center=True)[0, 0]
        Qoo = np.sqrt(1/muOO/freqoo)
        Qoh = np.sqrt(1/muOH/freqoh)
        fancyF = FrrR * Qoh**2 * Qoo
        return Constants.convert(fancyF, "wavenumbers", to_AU=False)

    def cubicharmonic(self):
        if self.FancyF is None:
            fancyF = self.find_FancyF()
            print(fancyF)
        else:
            fancyF = self.FancyF
        deltaQ = fancyF / (2*self.omegaOO)
        intensities = np.zeros(3)
        energies = np.zeros(3)
        factorial = [1, 1, 2]
        for i in np.arange(3):
            energies[i] = self.omegaOH - (fancyF**2/(8*self.omegaOO)) + (self.omegaOO*i)
            numer = np.exp(-1*deltaQ**2/2)*deltaQ**(2*i)
            denom = 2**i*factorial[i]
            intensities[i] = numer / denom

        return energies, intensities

class ModelAnharmonic:
    def __init__(self, moleculeObj, CC=False):
        self.molecule = moleculeObj
        if self.molecule.MoleculeName == "H9O4pls":
            self.cluster = "tetramer"
        elif self.molecule.MoleculeName == "H7O3pls":
            self.cluster = "trimer"
        else:
            raise Exception(f"Can not calculate Anharmonic Model Potential for {self.molecule.MoleculeName}")
        self.CC = CC

    def anharmonicmodelpotential(self):
        import os
        dvr_dir = os.path.join(self.molecule.mol_dir, "DVR Results")
        if self.CC:
            data = np.loadtxt(f"{dvr_dir}/spect_{self.cluster}_fancyF.dat", skiprows=1)
        else:
            data = np.loadtxt(f"{dvr_dir}/spect_{self.cluster}.dat", skiprows=1)
        ens = data[:, 1]
        matel = data[:, 5]
        idx = np.argwhere(data[:, 7] > 0.2)
        id = idx[:3].flatten()
        energies = ens[id]
        matrixEl = matel[id]
        return energies, matrixEl


class ModelHarmonic:
    # calculates a model potential with or without cubic coupling based off of XH/OO grid. Must put in "2D" molObj
    def __init__(self, moleculeObj, CC=False):
        self.molecule = moleculeObj
        self.CC = CC
        self.energy_array = self.molecule.logData.energies
        self._massdict = None
        self._finitevals = None
        self._force_constants = None
        self._grid = None
        self._CubicCoupling = None

    @property
    def massdict(self):
        if self._massdict is None:
            self._massdict = self.get_reducedmass()
        return self._massdict

    @property
    def finite_vals(self):
        if self._finitevals is None:
            self._finitevals = self.get_finitevals()
        return self._finitevals

    @property
    def force_constants(self):
        if self._force_constants is None:
            self._force_constants = self.getFC()  # Frr, FRR
        return self._force_constants

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self.get_grid()
        return self._grid

    @property
    def CubicCoupling(self):
        if self._CubicCoupling is None:
            self._CubicCoupling = self.getCC()
        return self._CubicCoupling

    def get_reducedmass(self):
        from Converter import Constants
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        massdict = dict()
        muOOH = ((2 * mO) * mH) / ((2 * mO) + mH)
        massdict["muOOH"] = muOOH
        muOH = 1/(1/mO + 1/mH)
        massdict["muOH"] = muOH
        muOO = mO / 2
        massdict["muOO"] = muOO
        return massdict

    def get_finitevals(self):
        from Converter import Constants
        import os
        # load in finite difference scan data and compute Force constants in OH and OO
        fs_dir = os.path.join(self.molecule.mol_dir, "Finite Scan Data")
        if self.molecule.MoleculeName == "H9O4pls":
            finite_vals = np.loadtxt(f"{fs_dir}/2D_finiteSPEtet_01_008.dat", skiprows=7)
        elif self.molecule.MoleculeName == "H7O3pls":
            finite_vals = np.loadtxt(f"{fs_dir}/2D_finiteSPEtri_01_008.dat", skiprows=7)
        else:
            raise Exception("Can't compute Force Constants of that molecule")
        finite_vals = finite_vals[:, 2:]
        finite_vals[:, 0] *= 2  # multiply OO/2 by 2 for OO values
        finite_vals[:, :2] = Constants.convert(finite_vals[:, :2], "angstroms", to_AU=True)  # convert OO/OH to bohr
        finite_vals[:, 2] -= min(finite_vals[:, 2])  # shift minimum to 0 so that energies match
        return finite_vals

    def getFC(self):
        from McUtils.Zachary import finite_difference
        # calculate harmonic force constants
        OO_FD = self.finite_vals[10:15, :]
        OH_FD = self.finite_vals[2::5, :]
        Frr = finite_difference(OO_FD[:, 1]-OO_FD[2, 1], OO_FD[:, 2], 2,
                                end_point_precision=0, stencil=5, only_center=True)[0]
        FRR = finite_difference(OH_FD[:, 0]-OH_FD[2, 0], OH_FD[:, 2], 2,
                                end_point_precision=0, stencil=5, only_center=True)[0]
        return Frr, FRR

    def get_grid(self):
        from Converter import Constants
        OOs = np.unique(self.energy_array[:, 0])
        OHs = np.unique(self.energy_array[:, 1])
        self.energy_array[:, 2] -= min(self.energy_array[:, 2])
        squarepot = np.reshape(self.energy_array[:, 2], (len(OOs), len(OHs)))
        mini = np.unravel_index(np.argmin(squarepot, axis=None), squarepot.shape)
        OOs_bohr = Constants.convert(OOs, "angstroms", to_AU=True)  # convert OO/OH to bohr for math
        XHs_bohr = Constants.convert(OHs, "angstroms", to_AU=True)
        print(OOs[mini[0]], OHs[mini[1]])
        # calculate Delta roh and Delta Roo
        DROO = OOs_bohr - OOs_bohr[mini[0]]
        Drxh = XHs_bohr - XHs_bohr[mini[1]]
        DROO_array, Drxh_array = np.meshgrid(DROO, Drxh)  # indexing='ij'
        # Droh_array = DROO_array / 2 - Drxh_array
        return DROO_array, Drxh_array

    def getCC(self):
        from McUtils.Zachary import finite_difference
        FDgrid = np.array(
            np.meshgrid(np.unique(self.finite_vals[:, 0]), np.unique(self.finite_vals[:, 1]))).T  # create mesh OO/OH
        FDvalues = np.reshape(self.finite_vals[:, 2], (5, 5))
        FancyF3 = finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5), accuracy=0, end_point_precision=0,
                                    only_center=True)[0, 0]
        return (1/2) * FancyF3

    def printFreqs(self):
        from Converter import Constants
        Frr = self.force_constants[0]
        FRR = self.force_constants[1]
        print("OO:", Constants.convert(np.sqrt(FRR/self.massdict["muOO"]), "wavenumbers", to_AU=False))
        print("OH:", Constants.convert(np.sqrt(Frr/self.massdict["muOH"]), "wavenumbers", to_AU=False))
        print("OOH:", Constants.convert(np.sqrt(Frr/self.massdict["muOOH"]), "wavenumbers", to_AU=False))
        print("ZPE:", Constants.convert((np.sqrt(FRR/self.massdict["muOO"]))/2+(np.sqrt(Frr/self.massdict["muOOH"]))/2,
                                        "wavenumbers", to_AU=False))

    def ModelHarmonicPotential(self, plotV=None):
        Frr = self.force_constants[0]
        FRR = self.force_constants[1]
        xh_HO = 1/2*Frr*(self.grid[1]**2) - 1/2*Frr*(self.grid[0]*self.grid[1])
        OO_HO = 1/2*(FRR+(1/4)*Frr)*(self.grid[0]**2)
        if self.CC:
            V_HO = OO_HO + xh_HO + (self.CubicCoupling * self.grid[1]**2 * self.grid[0])
        else:
            V_HO = OO_HO + xh_HO

        if plotV is not None:
            from Converter import Constants
            import matplotlib.pyplot as plt
            V_HOwave = Constants.convert(V_HO, "wavenumbers", to_AU=False)
            V_HOwave[V_HOwave > 25000] = 25000
            plt.contour(self.grid[0], self.grid[1], V_HOwave, colors='k', levels=10)
            plt.contourf(self.grid[0], self.grid[1], V_HOwave, cmap='Purples_r', levels=10)
            plt.colorbar()
            plt.xlabel("Delta OO")
            plt.ylabel("Delta OH")
            plt.savefig(plotV)
            plt.close()
        return V_HO

    def run_2D_DVR(self):
        """Runs 2D DVR over the original 2D potential should take flat xy array, and flat 2D grid in ATOMIC UNITS"""
        from PyDVR import DVR, ResultsInterpreter
        from Converter import Constants
        import os
        dvr_2D = DVR("ColbertMillerND")
        xy = np.column_stack((self.grid[0].flatten(), self.grid[1].flatten()))
        ens = self.ModelHarmonicPotential().flatten()
        res = dvr_2D.run(potential_grid=np.column_stack((xy, ens)),
                         divs=(100, 100), mass=[self.massdict["muOO"], self.massdict["muOOH"]], num_wfns=15,
                         domain=((min(xy[:, 0]), max(xy[:, 0])), (min(xy[:, 1]), max(xy[:, 1]))),
                         results_class=ResultsInterpreter)
        dvr_grid = Constants.convert(res.grid, "angstroms", to_AU=False)
        dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
        all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
        ResultsInterpreter.wfn_contours(res)
        oh1oo0 = int(input("OH=1 OO=0 Wavefunction Index: "))
        oh1oo1 = int(input("OH=1 OO=1 Wavefunction Index: "))
        oh1oo2 = int(input("OH=1 OO=2 Wavefunction Index: "))
        ens = np.zeros(4)
        wfns = np.zeros((4, res.wavefunctions[0].data.shape[0]))
        for i, wf in enumerate(res.wavefunctions):
            wfn = wf.data
            if i == 0:
                wfns[0] = wfn
                ens[0] = all_ens[i]
            elif i == oh1oo0:
                wfns[1] = wfn
                ens[1] = all_ens[i]
            elif i == oh1oo1:
                wfns[2] = wfn
                ens[2] = all_ens[i]
            elif i == oh1oo2:
                wfns[3] = wfn
                ens[3] = all_ens[i]
            else:
                pass
        # data saved in wavenumbers/angstrom
        dvr_dir = os.path.join(self.molecule.mol_dir, "DVR Results")
        if self.CC:
            npz_filename = f"{dvr_dir}/HMP_wCC_2D_DVR_OHOO.npz"
        else:
            npz_filename = f"{dvr_dir}/HMP_2D_DVR_OHOO.npz"
        np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot], vrwfn_idx=[0, oh1oo0, oh1oo1, oh1oo2],
                 energy_array=ens, wfns_array=wfns)
        return npz_filename
