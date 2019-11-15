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
        self._logData = None
        self._OHmass = None

    @property
    def logData(self):
        if self._logData is None:
            from GaussianHandler import LogInterpreter
            if self.method == "rigid":
                optBool = False
            else:
                optBool = True
            self._logData = LogInterpreter(*self.molecule.scanLogs, method=self.method, optimized=optBool,
                                           scancoords=self.scanCoords)
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
    
    def run_harOH_DVR(self):
        """ Runs a harmonic approximation using DVR to be fast over the OH coordinate at every OO value."""
        from PyDVR import DVR
        from McUtils.Zachary import finite_difference
        dvr_1D = DVR("ColbertMiller1D")
        finite_dict = self.logData.finite_dict(midpoint=True)
        roos = np.array(list(finite_dict.keys()))
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
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=True)
            ens = Constants.convert((res.wavefunctions.energies + min(y)), "wavenumbers", to_AU=True)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        oh_pots = np.column_stack((roos, energies_array[:, :4]))
        return potential, oh_pots, wavefunctions_array

    def run_anharOH_DVR(self):
        """ Runs anharmonic DVR over the OH coordinate at every OO value."""
        from PyDVR import DVR
        from PotentialHandlers import Potentials1D
        dvr_1D = DVR("ColbertMiller1D")
        cut_dict = self.logData.cut_dictionary(midpoint=True)
        roos = np.array(list(cut_dict.keys()))
        energies_array = np.zeros((len(cut_dict), self.desiredEnergies))
        wavefunctions_array = np.zeros((len(cut_dict), self.NumPts, self.desiredEnergies))

        for j, n in enumerate(cut_dict):
            x = Constants.convert(cut_dict[n][:, 0], "angstroms", to_AU=True)
            mini = min(x) - 0.3
            maxi = max(x) + 0.3
            en = cut_dict[n][:, 1]
            res = dvr_1D.run(potential_function=Potentials1D().potlint(x, en), mass=self.OHmass,
                             divs=self.NumPts, domain=(mini, maxi), num_wfns=self.desiredEnergies)
            potential = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=True)
            ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=True)
            energies_array[j, :] = ens
            wavefunctions_array[j, :, :] = res.wavefunctions.wavefunctions
        oh_pots = np.column_stack((roos, energies_array[:, :4]))
        return potential, oh_pots, wavefunctions_array

