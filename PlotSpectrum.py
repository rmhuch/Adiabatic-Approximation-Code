import numpy as np
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, moleculeObj=None, spectType=None, OODVRnpz=None, OHDVRnpz=None, DVRmethod=None):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.OODVRnpz = OODVRnpz
        self.OHDVRnpz = OHDVRnpz
        self.spectType = spectType
        if self.spectType == "Franck-Condon":
            self.tmObj = None
            self.shSpectType = "FC"
        elif self.spectType == "Transition Dipole Moment":
            from transitionmoment import TransitionMoment
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
            self.shSpectType = "TDM"
        elif self.spectType == "2D w/TDM":
            # should pass option for loading in 2D class and plotting that spectrum
            pass
        else:
            raise Exception("Unknown Spectrum Type")
        self.OODVRres = np.load(OODVRnpz)
        self.DVRmethod = DVRmethod
        self._intensities = None

    @property
    def intensities(self):
        if self._intensities is None:
            self._intensities = self.gettingIntense()
        return self._intensities

    def gettingIntense(self):
        intents = np.zeros(3)
        ooWfns = self.OODVRres["wfns_array"]
        wfnAmpIdx = np.argwhere(ooWfns[0, :, 0] > 1E-5)
        gs_wfn = ooWfns[0, wfnAmpIdx, 0]
        for i in np.arange(3):  # excited state wfn
            if self.tmObj is not None:
                trans_mom = self.tmObj.mus[1][:, 1:]
                comp_intents = np.zeros(3)
                for j in np.arange(3):  # transition moment component
                    es_wfn = ooWfns[1, wfnAmpIdx, i].T
                    super_es = trans_mom[:, j].T * es_wfn
                    comp_intents[j] = np.dot(gs_wfn.T, super_es.T)
                intents[i] = np.linalg.norm(comp_intents) ** 2
            else:
                es_wfn = ooWfns[1, wfnAmpIdx, i]
                intents[i] = np.dot(gs_wfn.T, es_wfn) ** 2
        return intents

    def make_spect(self, normalize=True, invert=False, line_type='b-', freq_shift=0):
        freqs = self.OODVRres["energy_array"][1, :3] - self.OODVRres["energy_array"][0, 0]
        intents = self.intensities * freqs
        norm_intents = intents / np.sum(intents)
        with open(f"{self.molecule.method}_{self.DVRmethod}OH_{self.shSpectType}spectrum.dat", "w") as f:
            f.write(f"{self.molecule.method} scan {self.DVRmethod} OH {self.spectType} Spectrum Values \n")
            f.write(f"Frequencies: {freqs} \n")
            f.write(f"Intensity (no Freq): {np.sum(self.intensities)} \n")
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
        markerline, stemline, baseline = plt.stem(frequency, intensity,
                                                  linefmt=line_type, markerfmt=' ', use_line_collection=True,
                                                  label=f"{self.molecule.method} {self.DVRmethod} OH {self.spectType}")
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        plt.ylim(0, 1)
        # plt.xlim(2200, 3700)
        plt.ylabel('Intensity')
        plt.xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        # return fig



