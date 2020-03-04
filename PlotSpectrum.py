import numpy as np
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, moleculeObj=None, spectType=None, TDMtype=None, OODVRnpz=None, OHDVRnpz=None, TwoDnpz=None, DVRmethod=None):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.spectType = spectType
        self.TDMtype = TDMtype
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
            if self.shSpectType == "2DTDM":
                self._intensities = self.getting2DIntense()
            else:
                self._intensities = self.gettingIntense()
        return self._intensities

    def find_FancyF(self, woo=None, woh=None):
        import os
        from Converter import Constants
        from McUtils.Zachary import finite_difference
        mO = Constants.mass("O", to_AU=True)
        mH = Constants.mass("H", to_AU=True)
        freqoo = Constants.convert(woo, "wavenumbers", to_AU=True)
        muOO = mO / 2
        freqoh = Constants.convert(woh, "wavenumbers", to_AU=True)
        muOH = ((2 * mO) * mH) / ((2 * mO) + mH)
        FD_file = os.path.join(self.molecule.mol_dir, 'Finite Scan Data', "newRigid_2D_finiteData.dat")
        finite_vals = np.loadtxt(FD_file)  # Energy OO OH
        finite_vals[:, 0] *= 2
        finite_vals[:, :2] = Constants.convert(finite_vals[:, :2], "angstroms", to_AU=True)  # convert to bohr for math
        idx = np.lexsort((finite_vals[:, 0], finite_vals[:, 1]))  # resort so same oh different oo
        finite_vals = finite_vals[idx]
        finite_vals = finite_vals.reshape((5, 5, 3))
        FR = np.zeros(5)
        # compute first derivative wrt oo FR
        for j in range(5):
            x = finite_vals[j, :, 0]  # roos
            y = finite_vals[j, :, 2]  # energies
            FR[j] = finite_difference(x, y, 1, end_point_precision=0, stencil=5, only_center=True)[0]
        print(f"FR: {FR}")
        # compute mixed derivative FrrR
        FrrR = finite_difference(finite_vals[:, 1, 1], FR, 2, end_point_precision=0, stencil=5, only_center=True)[0]
        print(f"FrrR: {FrrR}")
        Qoo = np.sqrt(1/muOO/freqoo)
        Qoh = np.sqrt(1/muOH/freqoh)
        fancyF = FrrR * Qoh**2 * Qoo
        return Constants.convert(fancyF, "wavenumbers", to_AU=False)

    def cubicharmonic(self, woo, woh, fancyF=None, gsEoh=0, color=None, label=""):
        if fancyF is None:
            fancyF = self.find_FancyF(woo=woo, woh=woh)
            print(fancyF)
        else:
            fancyF = fancyF
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
        fig = plt.gcf()
        fig.set_size_inches(8, 4)
        plt.rcParams.update({'font.size': 16})
        markerline, stemline, baseline = plt.stem(frequencies, norm_intents,
                                                  linefmt=color, markerfmt=' ', use_line_collection=True,
                                                  label=f"Cubic Harmonic {label}")
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        # plt.ylim(0, 1)

    def getting2DIntense(self):
        from IntensityCalculator import Intensities
        twoDres = np.load(self.TwoDnpz)
        twoDwfns = twoDres["wfns_array"]
        tdms = self.tmObj.TwoDtdms[1]
        if self.TDMtype == "Dipole Surface":
            trans_mom = tdms["poly"]
        elif self.TDMtype == "Cubic":
            trans_mom = tdms["cubic"]
        elif self.TDMtype == "Quadratic":
            trans_mom = tdms["quad"]
        elif self.TDMtype == "Quadratic OH only":
            trans_mom = tdms["quadOH"]
        elif self.TDMtype == "Quadratic Bilinear":
            trans_mom = tdms["quadbilin"]
        elif self.TDMtype == "Linear":
            trans_mom = tdms["lin"]
        elif self.TDMtype == "Linear OH only":
            trans_mom = tdms["const"]
        else:
            raise Exception("Can't determine TDM type.")
        intensities = Intensities.TwoD(twoDwfns, trans_mom)
        return intensities

    def gettingIntense(self):
        from IntensityCalculator import Intensities
        # rewrite this so that it calls to intensity class which either returns FC intensities, tdm intensities,
        # or expanded tdm intensities.
        OODVRres = np.load(self.OODVRnpz)
        ooWfns = OODVRres["wfns_array"]
        gs_wfn = ooWfns[0, :, 0]
        es_wfn = ooWfns[1, :, 0:3]
        tdms = self.tmObj.tdms[1]
        if self.tmObj is not None:  # TDM
            if self.TDMtype == "Poly":
                trans_mom = tdms["poly"]
            elif self.TDMtype == "Cubic":
                trans_mom = tdms["cubic"]
            elif self.TDMtype == "Quadratic":
                trans_mom = tdms["quad"]
            elif self.TDMtype == "Linear":
                trans_mom = tdms["lin"]
            elif self.TDMtype == "Constant":
                trans_mom = tdms["const"]
            else:
                raise Exception("Can't determine TDM type.")
            intents = Intensities.TDM(gs_wfn, es_wfn, trans_mom)
        else:  # FC
            intents = Intensities.FranckCondon(gs_wfn, es_wfn)
        return intents

    def make_spect(self, normalize=True, invert=False, line_type='b-', freq_shift=0, savefile=False):
        if self.TDMtype is None:
            TDMtype = ""
        else:
            TDMtype = self.TDMtype
        if self.spectType == "2D w/TDM":
            twoDres = np.load(self.TwoDnpz)
            filename = f"{self.molecule.method}_{TDMtype}{self.shSpectType}spectrum.dat"
            title = f"{self.molecule.method} scan {self.spectType} Spectrum Values: "
            freqs = twoDres["energy_array"][1:] - twoDres["energy_array"][0]
        else:
            filename = f"{self.molecule.method}_{self.DVRmethod}OH_{TDMtype}{self.shSpectType}spectrum.dat"
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
        # fig = plt.gcf()
        # fig.set_size_inches(7.5, 2)
        plt.rcParams.update({'font.size': 20})

        if self.DVRmethod is None:
            markerline, stemline, baseline = plt.stem(frequency, intensity,
                                                      linefmt=line_type, markerfmt=' ', use_line_collection=True,
                                                      label=f"{self.molecule.method} {TDMtype} {self.spectType}")
        else:
            markerline, stemline, baseline = plt.stem(frequency, intensity, linefmt=line_type, markerfmt=' ',
                                                      use_line_collection=True, label=
                                                      f"{self.molecule.method} {self.DVRmethod} OH {TDMtype} {self.spectType}")
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        plt.ylim(0, 1)
        # plt.xlim(1800, 3200)
        plt.ylabel('Intensity')
        plt.xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        # return fig



