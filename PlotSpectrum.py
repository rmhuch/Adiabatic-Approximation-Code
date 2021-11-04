import numpy as np

class Spectrum:
    def __init__(self, moleculeObj=None, spectType=None, TDMtype=None, adiabatType=None, CHobj=None, AMPobj=None,
                 OODVRnpz=None, OHDVRnpz=None, TwoDnpz=None):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.spectType = spectType
        self.TDMtype = TDMtype
        if self.spectType == "Franck-Condon":
            self.OODVRnpz = OODVRnpz
            self.OHDVRnpz = OHDVRnpz
            self.DVRmethod = adiabatType
            self.tmObj = None
            self.shSpectType = "FC"

        elif self.spectType == "Transition Dipole Moment":
            from transitionmoment import TransitionMoment
            self.OODVRnpz = OODVRnpz
            self.OHDVRnpz = OHDVRnpz
            self.TwoDnpz = TwoDnpz
            self.DVRmethod = adiabatType
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D", TwoDnpz=self.TwoDnpz,
                                          OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
            self.shSpectType = "TDM"

        elif self.spectType == "2D w/TDM":
            self.shSpectType = "2DTDM"
            from transitionmoment import TransitionMoment
            self.TwoDnpz = TwoDnpz
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="2D", TwoDnpz=self.TwoDnpz)
            self.DVRmethod = None

        elif self.spectType == "Harmonic Model":
            self.shSpectType = "harmModel"
            from transitionmoment import TransitionMoment
            self.TwoDnpz = TwoDnpz
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="2D", TwoDnpz=self.TwoDnpz) # , delta=True)
            self.DVRmethod = None

        elif self.spectType == "Harmonic Model w/CC":
            self.shSpectType = "harmModelCC"
            from transitionmoment import TransitionMoment
            self.TwoDnpz = TwoDnpz
            self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="2D", TwoDnpz=self.TwoDnpz, delta=True)
            self.DVRmethod = None

        elif self.spectType == "Cubic Harmonic":
            self.shSpectType = "cubicHarm"
            self.CHobj = CHobj
            self.tmObj = None
            self.DVRmethod = None

        elif self.spectType == "Anharmonic Model":
            self.shSpectType = "anharmModel"
            self.AMPobj = AMPobj
            self.tmObj = None
            self.DVRmethod = None

        elif self.spectType == "Anharmonic Model w/CC":
            self.shSpectType = "anharmModelCC"
            self.AMPobj = AMPobj
            self.tmObj = None
            self.DVRmethod = None

        else:
            raise Exception("Unknown Spectrum Type")
        self._intensities = None
        self._valuesDict = None

    @property
    def intensities(self):
        if self._intensities is None:
            if self.shSpectType == "cubicHarm":
                self._intensities = self.CHobj.cubicharmonic()  # returns energies then intensities
            elif self.shSpectType == "anharmModel":
                self._intensities = self.AMPobj.anharmonicmodelpotential()  # returns energies then matrix elements
            elif self.shSpectType == "anharmModelCC":
                self._intensities = self.AMPobj.anharmonicmodelpotential()
            elif self.shSpectType == "2DTDM":
                self._intensities = self.getting2DIntense()
            elif self.shSpectType == "harmModel" or self.shSpectType == "harmModelCC":
                self._intensities = self.getting2DIntense()
            else:
                self._intensities = self.gettingIntense()
        return self._intensities

    @property
    def valuesDict(self):
        if self._valuesDict is None:
            self._valuesDict, fig = self.make_spect()
        return self._valuesDict

    def getting2DIntense(self):
        from IntensityCalculator import Intensities
        twoDres = np.load(self.TwoDnpz)
        twoDwfns = twoDres["wfns_array"]
        tdms = self.tmObj.TwoD_dms[1]
        if self.TDMtype == "Dipole Surface":
            trans_mom = tdms["dipSurf"]
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
            trans_mom = tdms["linOH"]
        else:
            raise Exception("Can't determine TDM type.")
        intensities = Intensities.TwoD(twoDwfns, trans_mom, gridpoints=twoDres["grid"])
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
            if self.TDMtype == "Dipole Surface":
                trans_mom = tdms["dipSurf"]
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
                trans_mom = tdms["linOH"]
            else:
                raise Exception("Can't determine TDM type.")
            intents = Intensities.TDM(gs_wfn, es_wfn, trans_mom)
        else:  # FC
            intents = Intensities.FranckCondon(gs_wfn, es_wfn)
        return intents

    def make_spect(self, normalize=True, invert=False, line_type='b-', freq_shift=0, fig=None, addLabel=None):
        import matplotlib.pyplot as plt
        if self.spectType == "Cubic Harmonic":
            title = "Cubic Harmonic Spectrum Values: "
            freqs = self.intensities[0]
            matEls = None
            intents = self.intensities[1]
        elif self.spectType == "Anharmonic Model" or self.spectType == "Anharmonic Model w/CC":
            title = f"{self.spectType} Potential Values: "
            freqs = self.intensities[0]
            matEls = self.intensities[1]
            intents = matEls * freqs
        elif self.spectType == "Harmonic Model" or self.spectType == "Harmonic Model w/CC":
            title = f"{self.spectType} {self.TDMtype} Values: "
            twoDres = np.load(self.TwoDnpz)
            freqs = twoDres["energy_array"][1:] - twoDres["energy_array"][0]
            matEls = self.intensities
            intents = self.intensities * freqs
        elif self.spectType == "2D w/TDM":
            twoDres = np.load(self.TwoDnpz)
            title = f"{self.spectType} {self.TDMtype} Spectrum Values: "
            freqs = twoDres["energy_array"][1:] - twoDres["energy_array"][0]
            matEls = self.intensities
            intents = self.intensities * freqs
        else:
            title = f"{self.DVRmethod} OH/OO {self.spectType} {self.TDMtype} Spectrum Values: "
            OODVRres = np.load(self.OODVRnpz)
            freqs = OODVRres["energy_array"][1, :3] - OODVRres["energy_array"][0, 0]
            matEls = self.intensities
            intents = self.intensities * freqs
        norm_intents = intents / np.sum(intents)

        if invert:
            intents *= -1
        if normalize:
            intensity = norm_intents
        else:
            intensity = intents
        if addLabel is None:
            if self.DVRmethod is not None:
                addLabel = self.DVRmethod
            else:
                addLabel = ""
        else:
            addLabel = addLabel
        if self.TDMtype is None:
            TDMstr = ""
        else:
            TDMstr = self.TDMtype
        label = f"{self.spectType} {TDMstr} {addLabel}"
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
        valuesDict = {"label": label, "title": title, "matrix elements": matEls, "norm_intensities": norm_intents,
                      "frequencies": freqs, "intensities": intents}
        return valuesDict, (fig, ax)



