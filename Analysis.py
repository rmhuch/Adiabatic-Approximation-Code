from Converter import Constants
import numpy as np


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
        self._OHmass = None

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
    def OHmass(self):
        if self._OHmass is None:
            self._OHmass = self.get_reducedmass()
        return self._OHmass

    def get_reducedmass(self):
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        mD = Constants.mass("D", to_AU=True)
        if self.sharedProt == 'H':
            muOH = ((2 * mO) * mH) / ((2 * mO) + mH)
            mu = muOH
        elif self.sharedProt == 'D':
            muOD = ((2 * mO) * mD) / ((2 * mO) + mD)
            mu = muOD
        else:
            raise Exception("No reduced mass calculated.")
        return mu

    @staticmethod
    def wfn_flipper(wavefunctions_array, plotPhasedWfns=False):
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
                if plotPhasedWfns:
                    x = np.arange(len(gs_wfn))
                    plt.plot(x, wavefunctions_array[k - 1, :, 0], "--k", x, wavefunctions_array[k - 1, :, 1], "--k")
                    plt.plot(x, wavefunctions_array[k - 1, :, 2], "--k", x, wavefunctions_array[k - 1, :, 3], "--k")
                    plt.plot(x, gs_wfn, x, es_wfn)
                    plt.plot(x, nes_wfn, x, nnes_wfn)
                    plt.show()
            wfns[k, :, 0] = gs_wfn
            wfns[k, :, 1] = es_wfn
            wfns[k, :, 2] = nes_wfn
            wfns[k, :, 3] = nnes_wfn
        return wfns
    
    def run_harOH_DVR(self, plotPhasedWfns=None):
        """ Runs a harmonic approximation using DVR to be fast over the OH coordinate at every OO value."""
        import os
        from PyDVR import DVR
        from McUtils.Zachary import finite_difference
        dvr_1D = DVR("ColbertMiller1D")
        finite_dict = self.logData.finite_dict(midpoint=True)
        roos = np.array(list(finite_dict.keys()))
        potential_array = np.zeros((len(finite_dict), self.NumPts, 2))
        energies_array = np.zeros((len(finite_dict), self.desiredEnergies))
        wavefunctions_array = np.zeros((len(finite_dict), self.NumPts, self.desiredEnergies))

        for j, n in enumerate(finite_dict):
            x = Constants.convert(finite_dict[n][:, 0], "angstroms", to_AU=True)
            sx = x - np.amin(x)
            y = finite_dict[n][:, 1]
            k = finite_difference(sx, y, 2, end_point_precision=0, stencil=5, only_center=True)[0]
            mini = min(sx) - 1.0
            maxi = max(sx) + 1.0
            res = dvr_1D.run(potential_function="harmonic_oscillator", k=k, mass=self.OHmass,
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            potential_array[j, :, 0] = Constants.convert(res.grid, "angstroms", to_AU=False)
            potential_array[j, :, 1] = potential
            ens = Constants.convert((res.wavefunctions.energies + min(y)), "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        epsilon_pots = np.column_stack((roos, energies_array[:, :4]))
        npz_filename = os.path.join(self.DVRdir, f"OH_harmDVR_numpts{self.NumPts}_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns)
        np.savez(npz_filename, method="harm", potential=potential_array, epsilonPots=epsilon_pots,
                 wfns_array=wavefuns_array)
        return npz_filename

    def run_anharOH_DVR(self, plotPhasedWfns=None):
        """ Runs anharmonic DVR over the OH coordinate at every OO value."""
        import os
        from PyDVR import DVR
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
            res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=self.OHmass,
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            potential_array[j, :, 0] = Constants.convert(res.grid, "angstroms", to_AU=False)
            potential_array[j, :, 1] = potential
            ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        epsilon_pots = np.column_stack((roos, energies_array[:, :4]))
        npz_filename = os.path.join(self.DVRdir, f"OH_anharmDVR_numpts{self.NumPts}_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns)
        np.savez(npz_filename, method="anharm", potential=potential_array, epsilonPots=epsilon_pots,
                 wfns_array=wavefuns_array)
        return npz_filename

    def run_OO_DVR(self, OHDVRres=None, plotPhasedWfns=None):
        """Runs OO DVR over the epsilon potentials"""
        import os
        from PyDVR import DVR
        from PotentialHandlers import Potentials1D
        dvr_1D = DVR("ColbertMiller1D")
        OHresults = np.load(OHDVRres)
        epsi_pots = OHresults["epsilonPots"]
        potential_array = np.zeros((2, self.NumPts, 2))
        energies_array = np.zeros((2, self.desiredEnergies))
        wavefunctions_array = np.zeros((2, self.NumPts, self.desiredEnergies))
        mO = Constants.mass("O", to_AU=True)
        muOO = mO / 2
        x = Constants.convert(epsi_pots[:, 0], "angstroms", to_AU=True)
        mini = min(x) - 0.3
        maxi = max(x) + 0.15
        for j in np.arange(2):
            en = Constants.convert(epsi_pots[:, j + 1], "wavenumbers", to_AU=True)
            res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=muOO,
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
            potential_array[j, :, 0] = Constants.convert(res.grid, "angstroms", to_AU=False)
            potential_array[j, :, 1] = potential
            ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        npz_filename = os.path.join(self.DVRdir,
                                    f"OODVR_w{OHresults['method']}OHDVR_numpts{self.NumPts}_energies{self.desiredEnergies}.npz")
        # data saved in wavenumbers/angstroms
        wavefuns_array = self.wfn_flipper(wavefunctions_array, plotPhasedWfns=plotPhasedWfns)
        np.savez(npz_filename, potential=potential_array, energy_array=energies_array, wfns_array=wavefuns_array)
        return npz_filename


class TransitionMoment:
    def __init__(self, moleculeObj=None, OHDVRnpz=None):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.method = self.molecule.method
        self.scanCoords = self.molecule.scanCoords
        self.OHDVRres = np.load(OHDVRnpz)
        self._logData = None
        self._embeddedCoords = None

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
        if self._embeddedCoords is None:
            self._embeddedCoords = self.embed()
        return self._embeddedCoords

    def embed(self):
        from Rotator import many_rotations
        newcoords = many_rotations(self.logData, centralO_atom=1, xAxis_atom=0, xyPlane_atom=None,
                                   outerO1=5, outerO2=8, inversion_atom=8)
        return newcoords

    def run_tm(self):
        """should be pased a sqaure grid of dipoles to embed and interpolate. call to transitionmoment.py
        and probably some embed.py class or somehing perhaps?"""
        import transitionmoment as tm
        from scipy import interpolate
        roos = self.embeddedCoords[:, 0, 0]
        dipadedodas = tm.interp_dipoles(self.embeddedCoords, self.OHDVRres)
        mus = tm.psi_trans(dipadedodas, self.OHDVRres)
        g = self.OHDVRres["potential"][0]
        interp_mus = np.zeros((3, len(g)))
        for i in np.arange(3):
            tck = interpolate.splrep(roos, mus[i, :], s=0)
            interp_mus[i, :] = interpolate.splev(g, tck, der=0)

