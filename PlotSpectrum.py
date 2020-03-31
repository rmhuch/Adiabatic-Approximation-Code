import numpy as np
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, moleculeObj=None, spectType=None, TDMtype=None, CHobj=None,
                 OODVRnpz=None, OHDVRnpz=None, TwoDnpz=None, DVRmethod=None):
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
            self.CHobj = CHobj

        else:
            raise Exception("Unknown Spectrum Type")
        self.DVRmethod = DVRmethod
        self._intensities = None

    @property
    def intensities(self):
        if self._intensities is None:
            if self.shSpectType == "cubicHarm":
                self._intensities = self.CHobj.cubicharmonic()  # returns energies then intensities
            elif self.shSpectType == "2DTDM":
                self._intensities = self.getting2DIntense()
            else:
                self._intensities = self.gettingIntense()
        return self._intensities

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
        if self.tmObj is not None:  # TDM
            tdms = self.tmObj.tdms[1]
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

    def make_spect(self, normalize=True, invert=False, line_type='b-', freq_shift=0, fig=None, addLabel=None):
        if self.spectType == "Cubic Harmonic":
            title = "Cubic Harmonic Spectrum Values: "
            freqs = self.intensities[0]
            intents = self.intensities[1]
            inents = intents
        elif self.spectType == "2D w/TDM":
            twoDres = np.load(self.TwoDnpz)
            title = f"{self.molecule.method} scan {self.spectType} Spectrum Values: "
            freqs = twoDres["energy_array"][1:] - twoDres["energy_array"][0]
            inents = self.intensities
            intents = self.intensities * freqs
        else:
            title = f"{self.molecule.method} scan {self.DVRmethod} OH {self.spectType} Spectrum Values: "
            OODVRres = np.load(self.OODVRnpz)
            freqs = OODVRres["energy_array"][1, :3] - OODVRres["energy_array"][0, 0]
            inents = self.intensities
            intents = self.intensities * freqs
        norm_intents = intents / np.sum(intents)

        if invert:
            intents *= -1
        if normalize:
            intensity = norm_intents
        else:
            intensity = intents
        if addLabel is None:
            addLabel = ""
        else:
            addLabel = addLabel
        label = f"{self.spectType} {self.TDMtype} {addLabel}"
        frequency = freqs + freq_shift
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig, ax = fig
        # fig.rcParams.update({'font.size': 20})
        markerline, stemline, baseline = ax.stem(frequency, intensity, linefmt=line_type, markerfmt=' ',
                                                 use_line_collection=True, label=label)
        plt.setp(stemline, 'linewidth', 6.0)
        plt.setp(baseline, visible=False)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Energy ($\mathrm{cm}^{-1}$)')
        plt.tight_layout()
        valuesDict = {"label": label, "title": title, "intensities": inents, "norm_intensities": norm_intents,
                      "frequencies": freqs}
        return valuesDict, (fig, ax)

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

