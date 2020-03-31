from MolecularSys import *
from AdiabaticAnalysis import *
from Figures import *
from PlotSpectrum import *


def makeMolecule(MolDirName, embedDict, scancoords=((0, 1), (1, 2)), method="rigid", dimension="1D"):
    if MolDirName == "H9O4pls":
        atomStr = ["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"]
    elif MolDirName == "H7O3pls":
        atomStr = ["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"]
    elif MolDirName == "H5O2pls":
        atomStr = ["O", "O", "H", "D", "D", "D", "D"]
    else:
        raise Exception("No atom list defined.")
    mol = Molecule(MoleculeName=MolDirName,
                   dimension=dimension,
                   atom_str=atomStr,
                   method=method,
                   scanCoords=scancoords,
                   embed_dict=embedDict)
    return mol


def runAdiabaticApprox(molObj, anharmonic=True, plotPhasedWfns=False, makePlots=True,
                       desiredenergies=4, numPts=500, ohwfns2plt=2, oowfns2plt=3):
    import os
    dvr_dir = os.path.join(molObj.mol_dir, "DVR Results")
    AAobj = AdiabaticApprox(moleculeObj=molObj,
                            DVR_desiredEnergies=desiredenergies,
                            NumPts=numPts)
    if anharmonic:
        AAobj.run_anharOH_DVR(plotPhasedWfns=plotPhasedWfns)
        ohdvr_filename = f"{dvr_dir}{molObj.method}_AnharmOHDVR_energies{desiredenergies}.npz"
        AAobj.run_OO_DVR(OHDVRres=ohdvr_filename, plotPhasedWfns=plotPhasedWfns)
        oodvr_filename = f"{dvr_dir}{molObj.method}_OODVR_wanharmOHDVR_energies{desiredenergies}.npz"
    else:
        AAobj.run_harOH_DVR(plotPhasedWfns=plotPhasedWfns)
        ohdvr_filename = f"{dvr_dir}{molObj.method}_HarmOHDVR_energies{desiredenergies}.npz"
        AAobj.run_OO_DVR(OHDVRres=ohdvr_filename, plotPhasedWfns=plotPhasedWfns)
        oodvr_filename = f"{dvr_dir}{molObj.method}_OODVR_wharmOHDVR_energies4.npz"
    if makePlots:
        PlotObj = AAplots(moleculeObj=molObj, OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
        PlotObj.ohWfn_plots(wfns2plt=ohwfns2plt)
        PlotObj.ooWfn_plots(wfns2plt=oowfns2plt)
        PlotObj.make_adiabatplots()

def runTMplots(molObj, anharmonic=True, numDVRenergies=4):
    dvr_dir = os.path.join(molObj.mol_dir, "DVR Results")
    if molObj.dimension == "1D":
        if anharmonic:
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_AnharmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wanharmOHDVR_energies{numDVRenergies}.npz"
        else:
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_HarmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wharmOHDVR_energies{numDVRenergies}.npz"
        tut = TMplots(moleculeObj=molObj, OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
        tut.DipoleSurfaces()
        # tut.TransitionMoments(ylim=(-0.5, 3))
        # tut.InterpolatedDips()
        # tut.componentTMs(ylim=(-0.5, 3))
    else:
        tot = TM2Dplots(moleculeObj=molObj,
                        TwoDnpz=f"{dvr_dir}/{molObj.method}_2D_DVR.npz")
        tot.DipoleSurfaces()
        tot.componentTMs()

def makeSpectSingle(molObj, spectType, lineType, CHobj=None, freq_shift=0, TDMtype=None, anharmonic=True, model=None,
                    invert=False, normalize=True, numDVRenergies=4, fig=None):
    # f"{self.molecule.method} {TDMtype} {self.spectType}"
    dvr_dir = os.path.join(molObj.mol_dir, "DVR Results")
    if molObj.dimension == "1D":
        twoDnpz = None
        label = None
        if anharmonic:
            dvr_method = "Anharmonic"
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_AnharmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wanharmOHDVR_energies{numDVRenergies}.npz"
        else:
            dvr_method = "Harmonic"
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_HarmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wharmOHDVR_energies{numDVRenergies}.npz"
    else:
        oodvr_filename = None
        ohdvr_filename = None
        dvr_method = None
        if model == "anharmCC":
            twoDnpz = "AOpot_2D_DVR.npz"
            label = "AO w/CC"
        elif model == "anharm":
            twoDnpz = "AOpot_2D_DVR_noCC.npz"
            label = "AO w/o CC"
        elif model == "harmCC":
            twoDnpz = "HOpot_2D_DVR.npz"
            label = "HO w/CC"
        elif model == "harm":
            twoDnpz = "HOpot_2D_DVR_noCC.npz"
            label = "HO w/o CC"
        else:
            twoDnpz = f"{dvr_dir}/{molObj.method}_2D_DVR.npz"
            label = None
    spectObj = Spectrum(moleculeObj=molObj, CHobj=CHobj, spectType=spectType, TDMtype=TDMtype, TwoDnpz=twoDnpz,
                        OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename, DVRmethod=dvr_method)
    SpectValues, SpectFig = spectObj.make_spect(normalize=normalize, invert=invert,
                                                line_type=lineType, freq_shift=freq_shift, fig=fig, addLabel=label)
    print(molObj.MoleculeName)
    print(SpectValues)
    return SpectValues, SpectFig

def makeSpectMultiple(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, filename, anharmonics, CHobj=None,
                      inverts=False, normalize=True, numDVRenergies=4):
    if len(molObjs) == 1:
        molObjs = molObjs*len(spectTypes)
    if isinstance(inverts, bool):
        inverts = [inverts]*len(spectTypes)
    if isinstance(anharmonics, bool):
        anharmonics = [anharmonics]*len(spectTypes)
    # if isinstance(normalize, bool):
    #     normalize = [normalize] * len(spectTypes)
    # if isinstance(numDVRenergies, bool):
    #     numDVRenergies = [numDVRenergies] * len(spectTypes)
    SpectFig = None
    for molObj, spectType, lineType, freq_shift, TDMtype, anharmonic, invert in \
            zip(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, anharmonics, inverts):
        if spectType == "2D newModel":
            SpectValues, SpectFig = makeSpectSingle(molObj, "2D w/TDM", lineType, model="anharm", freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        elif spectType == "2D newModelCC":
            SpectValues, SpectFig = makeSpectSingle(molObj, "2D w/TDM", lineType, model="anharmCC", freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)

        elif spectType == "2D newModelHarm":
            SpectValues, SpectFig = makeSpectSingle(molObj, "2D w/TDM", lineType, model="harm",
                                                    freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        elif spectType == "2D newModelHarmCC":
            SpectValues, SpectFig = makeSpectSingle(molObj, "2D w/TDM", lineType, model="harmCC",
                                                    freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        elif spectType == "Cubic Harmonic":
            SpectValues, SpectFig = makeSpectSingle(molObj, spectType, lineType, CHobj=CHobj, freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        else:
            SpectValues, SpectFig = makeSpectSingle(molObj, spectType, lineType, freq_shift=freq_shift,
                                                    TDMtype=TDMtype, anharmonic=anharmonics, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
    if filename is not None:
        SpectFig[0].savefig(filename)
    else:
        SpectFig[0].show()

def makeSpectFile(molObj, SpectValues):
    if savefile:
        with open(filename, "w") as f:
            f.write(f"{title} \n")
            f.write(f"Frequencies: {freqs} \n")
            f.write(f"Intensities: {intents} \n")
            f.write(f"Intensity (w/Freq): {np.sum(intents)} \n")
            f.write(f"Normalized Intensities: {norm_intents} \n")

if __name__ == '__main__':
    tetEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "outerO1": 5, "outerO2": 8, "inversion_atom": 8}
    triEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "xyPlane_atom": 5, "inversion_atom": 9}
    diEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "inversion_atom": 3}

    tesst = makeMolecule("H5O2pls", diEmbedDict, dimension="2D")
    tesst1 = makeMolecule("H5O2pls", diEmbedDict, dimension="1D")
    tessst = makeMolecule("H7O3pls", triEmbedDict, dimension="2D")
    tessst1 = makeMolecule("H7O3pls", triEmbedDict, dimension="1D")
    tesssst = makeMolecule("H9O4pls", tetEmbedDict, dimension="2D")
    tesssst1 = makeMolecule("H9O4pls", tetEmbedDict, dimension="1D")

    chtest = CubicHarmonic(tesssst1, omegaOO=362.32, omegaOH=2909.44, FancyF=102)

    runTMplots(tesssst)
    # runTMplots(tessst)
    # runTMplots(tesst1)
    # runTMplots(tessst1)
    # ST = ["2D w/TDM"]
    # LT = ["C0-", "C6-", "C5-", "C8-", "C7-", "C4-", "C3-"]
    # LT1 = ["C0-", "C6-", "C5-", "C4-", "C3-"]
    # LT2 = ["C0-", "C5-", "C8-", "C7-"]
    # FS = [-30, -20, -10, 0, 10, 20, 30]
    # FS1 = [-20, -10, 0, 10, 20]
    # FS2 = [-15, -5, 5, 15]
    # TT = ["Dipole Surface", "Cubic", "Quadratic", "Quadratic OH only", "Quadratic Bilinear", "Linear", "Linear OH only"]
    # TT1 = ["Dipole Surface", "Cubic", "Quadratic", "Linear", "Linear OH only"]
    # TT2 = ["Dipole Surface", "Quadratic", "Quadratic OH only", "Quadratic Bilinear"]
    # fn2 = "di2DComponentSpectrum_all.png"
    # fn3 = "tri2DComponentSpectrum_all_zoom.png"
    # fn4 = "tet2DComponentSpectrum_all.png"
    # # makeSpectMultiple([tesst], ST*len(TT), LT, FS, TT, fn2, anharmonics=True)
    # # makeSpectMultiple([tessst], ST*len(TT), LT, FS, TT, fn3, anharmonics=True)
    # # makeSpectMultiple([tesssst], ST*len(TT), LT, FS, TT, fn4, anharmonics=True)
    # ST1D = ["Transition Dipole Moment"]
    # LT1D = ["C1-", "C6-", "C5-", "C4-", "C3-"]
    # FS1D = [-20, -10, 0, 10, 20]
    # TT1D = ["Poly", "Cubic", "Quadratic", "Linear", "Constant"]
    # fn1D2 = "di1DComponentSpectrum_all.png"
    # fn1D3 = "tri1DComponentSpecrum_all_zoom.png"
    # fn1D4 = "tet1DComponentSpectrum_all.png"
    # makeSpectMultiple([tesst1], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D2, anharmonics=True)
    # makeSpectMultiple([tessst1], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D3, anharmonics=True)
    # makeSpectMultiple([tesssst1], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D4, anharmonics=True)

    molObjs = [tesssst1, tesssst]
    ST = ["Cubic Harmonic", "2D w/TDM"]
    LT = ["C9-", "C0-"]
    FS = [0]*len(ST)
    TT = [None, "Dipole Surface"]
    fn = "CH2D_Spectrum.png"
    makeSpectMultiple(molObjs, ST, LT, FS, TT, fn, anharmonics=True, CHobj=chtest)
       

