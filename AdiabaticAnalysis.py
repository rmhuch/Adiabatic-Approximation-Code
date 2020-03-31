from Converter import Constants
from PyDVR.DVR import *


class AdiabaticApprox:
    def __init__(self, moleculeObj=None, DVR_desiredEnergies=None, NumPts=None, sharedProt="H"):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.method = self.molecule.method
        self.scanCoords = self.molecule.scanCoords
        self.sharedProt = sharedProt
        self.desiredEnergies = DVR_desiredEnergies
        self.NumPts = NumPts
        self._DVRdir = None
        self._logData = None
        self._massdict = None

    @property
    def DVRdir(self):
        if self._DVRdir is None:
            import os
            self._DVRdir = os.path.join(self.molecule.mol_dir, "DVR Results")
        return self._DVRdir

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
    def massdict(self):
        if self._massdict is None:
            self._massdict = self.get_reducedmass()
        return self._massdict

    def get_reducedmass(self):
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        mD = Constants.mass("D", to_AU=True)
        massdict = dict()
        muOOH = ((2 * mO) * mH) / ((2 * mO) + mH)
        massdict["muOOH"] = muOOH
        muOOD = ((2 * mO) * mD) / ((2 * mO) + mD)
        massdict["muOOD"] = muOOD
        muOH = 1/(1/mO + 1/mH)
        massdict["muOH"] = muOH
        muOO = mO / 2
        massdict["muOO"] = muOO
        return massdict

    @staticmethod
    def wfn_flipper(wavefunctions_array, plotPhasedWfns=False, pot_array=None):
        """ Rephases output wavefunctions such that they are phased to the same orientation as the one before"""
        import matplotlib.pyplot as plt
        wfns = np.zeros((len(wavefunctions_array), len(wavefunctions_array[0]), 4))
        for k in np.arange(len(wavefunctions_array)):
            gs_wfn = wavefunctions_array[k, :, 0]
            es_wfn = wavefunctions_array[k, :, 1]
            nes_wfn = wavefunctions_array[k, :, 2]
            nnes_wfn = wavefunctions_array[k, :, 3]
            # calculate overlaps to check for phase consistency
            if k >= 1:
                gs_ovn = np.dot(wavefunctions_array[k - 1, :, 0], gs_wfn)
                if gs_ovn <= 0:
                    gs_wfn *= -1
                es_ovn = np.dot(wavefunctions_array[k - 1, :, 1], es_wfn)
                if es_ovn <= 0:
                    es_wfn *= -1
                nes_ovn = np.dot(wavefunctions_array[k - 1, :, 2], nes_wfn)
                if nes_ovn <= 0:
                    nes_wfn *= -1
                nnes_ovn = np.dot(wavefunctions_array[k - 1, :, 3], nnes_wfn)
                if nnes_ovn <= 0:
                    nnes_wfn *= -1
            wfns[k, :, 0] = gs_wfn
            wfns[k, :, 1] = es_wfn
            wfns[k, :, 2] = nes_wfn
            wfns[k, :, 3] = nnes_wfn
        if plotPhasedWfns:
            for k in np.arange(len(wavefunctions_array)):
                x = pot_array[k, :, 0]
                plt.plot(x, wfns[k, :, 0])
                plt.plot(x, wfns[k, :, 1]+1)
                plt.plot(x, wfns[k, :, 2]+2)
                plt.plot(x, wfns[k, :, 3]+3)
        return wfns
    
    def run_harOH_DVR(self, plotPhasedWfns=None):
        """ Runs a harmonic approximation using DVR to be fast over the OH coordinate at every OO value."""
        from McUtils.Zachary import finite_difference
        dvr_1D = DVR("ColbertMiller1D")
        finite_dict = self.logData.finite_dict(midpoint=True)
        roos = np.array(list(finite_dict.keys()))
        potential_array = np.zeros((len(finite_dict), self.NumPts, 2))
        energies_array = np.zeros((len(finite_dict), self.desiredEnergies))
        wavefunctions_array = np.zeros((len(finite_dict), self.NumPts, self.desiredEnergies))

        for j, n in enumerate(finite_dict):
            x = Constants.convert(finite_dict[n][:, 0], "angstroms", to_AU=True)
            min_idx = np.argmin(finite_dict[n][:, 1])
            sx = x - x[min_idx]
            y = finite_dict[n][:, 1]
            k = finite_difference(sx, y, 2, end_point_precision=0, stencil=5, only_center=True)[0]
            mini = min(sx) - 1.0
            maxi = max(sx) + 1.0
            res = dvr_1D.run(potential_function="harmonic_oscillator", k=k, mass=self.massdict["muOOH"],
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            grid = Constants.convert((res.grid + x[min_idx]), "angstroms", to_AU=False)
            shiftgrid = (n/2) + grid
            potential_array[j, :, 0] = shiftgrid
            potential_array[j, :, 1] = potential
            ens = Constants.convert((res.wavefunctions.energies + min(y)), "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        epsilon_pots = np.column_stack((roos, energies_array[:, :4]))
        npz_filename = os.path.join(self.DVRdir, f"{self.method}_HarmOHDVR_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms with potential shifted BACK to OH scan points
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns, pot_array=potential_array)
        np.savez(npz_filename, method="harm", potential=potential_array, epsilonPots=epsilon_pots,
                 wfns_array=wavefuns_array)
        return npz_filename

    def run_anharOH_DVR(self, plotPhasedWfns=None):
        """ Runs anharmonic DVR over the OH coordinate at every OO value."""
        from PotentialHandlers import Potentials1D
        dvr_1D = DVR("ColbertMiller1D")
        cut_dict = self.logData.cut_dictionary(midpoint=True)
        roos = np.array(list(cut_dict.keys()))
        potential_array = np.zeros((len(cut_dict), self.NumPts, 2))
        energies_array = np.zeros((len(cut_dict), self.desiredEnergies))
        wavefunctions_array = np.zeros((len(cut_dict), self.NumPts, self.desiredEnergies))

        for j, n in enumerate(cut_dict):
            x = Constants.convert(cut_dict[n][:, 0], "angstroms", to_AU=True)
            mini = min(x) - 0.3
            maxi = max(x) + 0.3
            en = cut_dict[n][:, 1]
            res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=self.massdict["muOOH"],
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            grid = Constants.convert(res.grid, "angstroms", to_AU=False)
            shiftgrid = (n/2) + grid
            potential_array[j, :, 0] = shiftgrid
            potential_array[j, :, 1] = potential
            ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        epsilon_pots = np.column_stack((roos, energies_array[:, :4]))
        npz_filename = os.path.join(self.DVRdir, f"{self.method}_AnharmOHDVR_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms with potential shifted BACK to OH scan points
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns, pot_array=potential_array)
        np.savez(npz_filename, method="anharm", potential=potential_array, epsilonPots=epsilon_pots,
                 wfns_array=wavefuns_array)
        return npz_filename

    def run_OO_DVR(self, OHDVRres=None, plotPhasedWfns=None):
        """Runs OO DVR over the epsilon potentials"""
        from PotentialHandlers import Potentials1D
        dvr_1D = DVR("ColbertMiller1D")
        OHresults = np.load(OHDVRres)
        epsi_pots = OHresults["epsilonPots"]
        potential_array = np.zeros((2, self.NumPts, 2))
        energies_array = np.zeros((2, self.desiredEnergies))
        wavefunctions_array = np.zeros((2, self.NumPts, self.desiredEnergies))
        x = Constants.convert(epsi_pots[:, 0], "angstroms", to_AU=True)
        mini = min(x) - 0.3
        maxi = max(x) + 0.15
        for j in np.arange(2):
            en = Constants.convert(epsi_pots[:, j + 1], "wavenumbers", to_AU=True)
            res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=self.massdict["muOO"],
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            potential_array[j, :, 0] = Constants.convert(res.grid, "angstroms", to_AU=False)
            potential_array[j, :, 1] = potential
            ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        npz_filename = os.path.join(self.DVRdir,
                                    f"{self.method}_OODVR_w{OHresults['method']}OHDVR_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns, pot_array=potential_array)
        np.savez(npz_filename, potential=potential_array, energy_array=energies_array, wfns_array=wavefuns_array)
        return npz_filename

    def run_2D_DVR(self):
        """Runs 2D DVR over the original 2D potential"""
        dvr_2D = DVR("ColbertMillerND")
        npz_filename = os.path.join(self.DVRdir, f"{self.method}_2D_DVR.npz")
        twoD_grid = self.logData.rawenergies
        xy = Constants.convert(twoD_grid[:, :2], "angstroms", to_AU=True)
        res = dvr_2D.run(potential_grid=np.column_stack((xy, twoD_grid[:, 2])),
                         divs=(100, 100), mass=[self.massdict["muOO"], self.massdict["muOH"]], num_wfns=15,
                         domain=((min(xy[:, 0]),  max(xy[:, 0])), (min(xy[:, 1]), max(xy[:, 1]))),
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
        # data saved in wavenumbers/angstroms
        np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot], vrwfn_idx=[0, oh1oo0, oh1oo1, oh1oo2],
                 energy_array=ens, wfns_array=wfns)
        return npz_filename

