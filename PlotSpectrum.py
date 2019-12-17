import numpy as np
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, moleculeObj=None, spectType=None, OODVRnpz=None, OHDVRnpz=None, TwoDnpz=None, DVRmethod=None):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.spectType = spectType
        if self.spectType == "Franck-Condon":
            self.OODVRnpz = OODVRnpz
            self.OHDVRnpz = OHDVRnpz
            self.tmObj = None
            self.shSpectType = "FC"

        elif self.spectType == "Transition Dipole Moment":
            from transitionmoment import TransitionMoment
            self.OODVRnpz = OODVRnpz
            self.OHDVRnpz = OHDVRnpz
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D",
                                          OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
            self.shSpectType = "TDM"

        elif self.spectType == "2D w/TDM":
            from transitionmoment import TransitionMoment
            self.TwoDnpz = TwoDnpz
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="2D", TwoDnpz=self.TwoDnpz)
            self.shSpectType = "2DTDM"

        elif self.spectType == "Cubic Harmonic":
            self.shSpectType = "cubicHarm"

        else:
            raise Exception("Unknown Spectrum Type")
        self.DVRmethod = DVRmethod
        self._intensities = None

    @property
    def intensities(self):
        if self._intensities is None:
            self._intensities = self.gettingIntense()
        return self._intensities

    def find_FancyF(self, woo, woh):
        from Converter import Constants
        from MolecularSys import MolecularOperations
        from McUtils.Zachary import finite_difference
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        freqoo = Constants.convert(woo, "wavenumbers", to_AU=True)
        muOO = mO / 2
        freqoh = Constants.convert(woh, "wavenumbers", to_AU=True)
        muOH = ((2 * mO) * mH) / ((2 * mO) + mH)
        molecularOpsObj = MolecularOperations(moleculeObj=self.molecule)
        finite_dict = molecularOpsObj.logData.finite_dict()
        roos = np.array(list(finite_dict.keys()))
        secondDerivs = np.zeros(len(roos))
        for j, n in enumerate(roos):
            x = Constants.convert(finite_dict[n][:, 0], "angstroms", to_AU=True)
            y = finite_dict[n][:, 1]
            secondDerivs[j] = finite_difference(x, y, 2, end_point_precision=0, stencil=5, only_center=True)[0]
        eq_idx = 5  # watch this.. think of better way to encode but for now this works (tet, tri)
        Qoo = np.sqrt(1/(muOO*freqoo))
        Qoh = 1/(muOH*freqoh)
        fancyF = Qoh * ((secondDerivs[eq_idx + 1] - secondDerivs[eq_idx - 1])/(2 * (roos[eq_idx] - roos[eq_idx - 1]))) * Qoo
        return Constants.convert(fancyF, "wavenumbers", to_AU=False)

    def cubicharmonic(self, woo, woh, gsEoh=0, color=None):
        fancyF = self.find_FancyF(woo=woo, woh=woh)
        deltaQ = fancyF / (2*woo)
        intensities = np.zeros(3)
        energies = np.zeros(3)
        factorial = [1, 1, 2]
        for i in np.arange(3):
            energies[i] = woh - (fancyF**2/(8*woo)) + (woo*i)
            numer = np.exp(-1*deltaQ**2/2)*deltaQ**(2*i)
            denom = 2**i*factorial[i]
            intensities[i] = numer / denom
        norm_intents = intensities / np.sum(intensities)
        frequencies = energies - gsEoh
        print(f"Cubic Harmonic Spectrum Values")
        print(f"Frequencies: {frequencies}")
        print(f"Intensity: {np.sum(intensities)}")
        print(f"Normalized Intensities: {norm_intents}")
        plt.rcParams.update({'font.size': 16})
        markerline, stemline, baseline = plt.stem(frequencies, norm_intents,
                                                  linefmt=color, markerfmt=' ', use_line_collection=True,
                                                  label=f"{self.molecule.method} Cubic Harmonic")
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        plt.ylim(0, 1)

    def gettingIntense(self):
        intents = np.zeros(3)
        if self.spectType == "2D w/TDM":
            twoDres = np.load(self.TwoDnpz)
            twoDwfns = twoDres["wfns_array"]
            dip_mat = self.tmObj.interp_2D_dipoles()
            for i in np.arange(1, len(twoDwfns)):
                x = 0
                for j in np.arange(3):
                    x += (np.dot(twoDwfns[0], (dip_mat[:, j] * twoDwfns[i]))) ** 2
                    # x is magnitude squared because instead of taking the sqrt and then squaring, I just didn't.
                intents[i-1] = x
        else:
            OODVRres = np.load(self.OODVRnpz)
            ooWfns = OODVRres["wfns_array"]
            wfnAmpIdx = np.argwhere(ooWfns[0, :, 0] > 1E-5)
            gs_wfn = ooWfns[0, wfnAmpIdx, 0]
            for i in np.arange(3):  # excited state wfn
                if self.tmObj is not None:  # TDM
                    trans_mom = self.tmObj.mus[1][:, 1:]
                    comp_intents = np.zeros(3)
                    for j in np.arange(3):  # transition moment component
                        es_wfn = ooWfns[1, wfnAmpIdx, i].T
                        super_es = trans_mom[:, j].T * es_wfn
                        comp_intents[j] = np.dot(gs_wfn.T, super_es.T)
                    intents[i] = np.linalg.norm(comp_intents) ** 2
                else:  # FC
                    es_wfn = ooWfns[1, wfnAmpIdx, i]
                    intents[i] = np.dot(gs_wfn.T, es_wfn) ** 2

        return intents

    def make_spect(self, normalize=True, invert=False, line_type='b-', freq_shift=0, savefile=False):
        if self.spectType == "2D w/TDM":
            twoDres = np.load(self.TwoDnpz)
            filename = f"{self.molecule.method}_{self.shSpectType}spectrum.dat"
            title = f"{self.molecule.method} scan {self.spectType} Spectrum Values: "
            freqs = twoDres["energy_array"][1:] - twoDres["energy_array"][0]
        else:
            filename = f"{self.molecule.method}_{self.DVRmethod}OH_{self.shSpectType}spectrum.dat"
            title = f"{self.molecule.method} scan {self.DVRmethod} OH {self.spectType} Spectrum Values: "
            OODVRres = np.load(self.OODVRnpz)
            freqs = OODVRres["energy_array"][1, :3] - OODVRres["energy_array"][0, 0]
        intents = self.intensities * freqs
        norm_intents = intents / np.sum(intents)
        # print(norm_intents)
        if savefile:
            with open(filename, "w") as f:
                f.write(f"{title} \n")
                f.write(f"Frequencies: {freqs} \n")
                f.write(f"Intensities: {intents} \n")
                f.write(f"Intensity (w/Freq): {np.sum(intents)} \n")
                f.write(f"Normalized Intensities: {norm_intents} \n")
        if invert:
            intents *= -1
        if normalize:
            intensity = norm_intents
        else:
            intensity = intents
        frequency = freqs + freq_shift
        # fig = plt.figure(figsize=(9, 6), dpi=300)
        plt.rcParams.update({'font.size': 16})
        if self.DVRmethod is None:
            markerline, stemline, baseline = plt.stem(frequency, intensity,
                                                      linefmt=line_type, markerfmt=' ', use_line_collection=True,
                                                      label=f"{self.molecule.method} {self.spectType}")
        else:
            markerline, stemline, baseline = plt.stem(frequency, intensity, linefmt=line_type, markerfmt=' ',
                                                      use_line_collection=True, label=
                                                      f"{self.molecule.method} {self.DVRmethod} OH {self.spectType}")
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        plt.ylim(0, 1)
        # plt.xlim(2200, 3700)
        plt.ylabel('Intensity')
        plt.xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        # return fig



