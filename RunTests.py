from MolecularSys import *
from AdiabaticAnalysis import *
from Figures import *
from PlotSpectrum import *
from PotExpansions import *


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


def runAdiabaticApprox(molObj, OH="anharmonic", OO="anharmonic", plotPhasedWfns=False, makePlots=True,
                       desiredenergies=4, numPts=500, ohwfns2plt=2, oowfns2plt=3):
    import os
    dvr_dir = os.path.join(molObj.mol_dir, "DVR Results")
    AAobj = AdiabaticApprox(moleculeObj=molObj,
                            DVR_desiredEnergies=desiredenergies,
                            NumPts=numPts)
    if molObj.dimension == "1D":
        if OH == "harmonic":
            AAobj.run_harOH_DVR(plotPhasedWfns=plotPhasedWfns)
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_HarmOHDVR_energies{desiredenergies}.npz"
            OHdvr = "harm"
        else:
            AAobj.run_anharOH_DVR(plotPhasedWfns=plotPhasedWfns)
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_AnharmOHDVR_energies{desiredenergies}.npz"
            OHdvr = "anharm"
        if OO == "harmonic":
            AAobj.run_harOO_DVR(OHDVRres=ohdvr_filename, plotPhasedWfns=plotPhasedWfns)
            oodvr_filename = f"{dvr_dir}/{molObj.method}_harmOODVR_w{OHdvr}OHDVR_energies4.npz"
        else:
            AAobj.run_OO_DVR(OHDVRres=ohdvr_filename, plotPhasedWfns=plotPhasedWfns)
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_w{OHdvr}OHDVR_energies{desiredenergies}.npz"
        if makePlots:
            PlotObj = AAplots(moleculeObj=molObj, OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
            # PlotObj.ohWfn_plots(wfns2plt=ohwfns2plt)
            PlotObj.ooWfn_plots(wfns2plt=oowfns2plt)
            PlotObj.make_adiabatplots()
    else:
        # AAobj.run_2D_DVR()
        AA2Dplots(molObj, f"{dvr_dir}/rigid_2D_DVR.npz").plotProjections()

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
        # tot.DipoleSurfaces()
        # tot.componentTMs()
        tot.plotDMcut()

def makeSpectSingle(molObj, spectType, lineType, CHobj=None, AMPobj=None, freq_shift=0, TDMtype=None,
                    adiabatType="anharmonic", invert=False, normalize=True, numDVRenergies=4, fig=None):
    # f"{self.molecule.method} {TDMtype} {self.spectType}"
    dvr_dir = os.path.join(molObj.mol_dir, "DVR Results")
    if "Harmonic Model" in spectType:
        oodvr_filename = None
        ohdvr_filename = None
        if "CC" in spectType:
            twoDnpz = f"{dvr_dir}/HMP_wCC_2D_DVR_OHOO.npz"
        else:
            twoDnpz = f"{dvr_dir}/HMP_2D_DVR_OHOO.npz"
    elif molObj.dimension == "1D":
        twoDnpz = None
        if adiabatType == "anharmonic":
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_AnharmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wanharmOHDVR_energies{numDVRenergies}.npz"
        elif adiabatType == "harmonic":
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_HarmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_harmOODVR_wharmOHDVR_energies{numDVRenergies}.npz"
        elif adiabatType == "anharm/harm":
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_AnharmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_harmOODVR_wanharmOHDVR_energies{numDVRenergies}.npz"
        elif adiabatType == "harm/anharm":
            ohdvr_filename = f"{dvr_dir}/{molObj.method}_HarmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}/{molObj.method}_OODVR_wharmOHDVR_energies{numDVRenergies}.npz"
        else:
            raise Exception(f"Hm.. I don't know the {adiabatType} adiabat Type..")
    else:
        oodvr_filename = None
        ohdvr_filename = None
        twoDnpz = f"{dvr_dir}/{molObj.method}_2D_DVR.npz"
    spectObj = Spectrum(moleculeObj=molObj, CHobj=CHobj, AMPobj=AMPobj, spectType=spectType, TDMtype=TDMtype,
                        adiabatType=adiabatType, TwoDnpz=twoDnpz, OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
    SpectValues, SpectFig = spectObj.make_spect(normalize=normalize, invert=invert,
                                                line_type=lineType, freq_shift=freq_shift, fig=fig)
    print(molObj.MoleculeName)
    print(SpectValues)
    return SpectValues, SpectFig

def makeSpectMultiple(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, filename, adiabatTypes,
                      inverts=False, normalize=True, numDVRenergies=4):
    if len(molObjs) == 1:
        molObjs = molObjs*len(spectTypes)
    if isinstance(inverts, bool):
        inverts = [inverts]*len(spectTypes)
    if len(adiabatTypes) == 1:
        adiabatTypes = adiabatTypes*len(spectTypes)
    SpectFig = None
    for molObj, spectType, lineType, freq_shift, TDMtype, adiabatType, invert in \
            zip(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, adiabatTypes, inverts):
        if spectType == "Cubic Harmonic":
            if molObj.MoleculeName == "H9O4pls":
                CHobj = CubicHarmonic(molObj, omegaOO=362.32, omegaOH=2909.44)
                # FancyF =
            elif molObj.MoleculeName == "H7O3pls":
                CHobj = CubicHarmonic(molObj, omegaOO=413.47, omegaOH=2525.51)
                # FancyF =
            else:
                raise Exception("No Cubic Harmonic approximation for this molecule.")
            SpectValues, SpectFig = makeSpectSingle(molObj, spectType, lineType, CHobj=CHobj, freq_shift=freq_shift,
                                                    TDMtype=TDMtype, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        elif "Anharmonic Model" in spectType:
            if "CC" in spectType:
                AMPobj = ModelAnharmonic(molObj, CC=True)
            else:
                AMPobj = ModelAnharmonic(molObj)
            SpectValues, SpectFig = makeSpectSingle(molObj, spectType, lineType, AMPobj=AMPobj, freq_shift=freq_shift,
                                                    TDMtype=TDMtype, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
        else:
            SpectValues, SpectFig = makeSpectSingle(molObj, spectType, lineType, freq_shift=freq_shift,
                                                    TDMtype=TDMtype, adiabatType=adiabatType, invert=invert,
                                                    normalize=normalize, numDVRenergies=numDVRenergies, fig=SpectFig)
    if filename is not None:
        SpectFig[0].savefig(filename)
    else:
        SpectFig[0].show()

# def makeSpectFile(molObj, SpectValues):
#     if savefile:
#         with open(filename, "w") as f:
#             f.write(f"{title} \n")
#             f.write(f"Frequencies: {freqs} \n")
#             f.write(f"Intensities: {intents} \n")
#             f.write(f"Intensity (w/Freq): {np.sum(intents)} \n")
#             f.write(f"Normalized Intensities: {norm_intents} \n")

if __name__ == '__main__':
    tetEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "outerO1": 5, "outerO2": 8, "inversion_atom": 8}
    triEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "xyPlane_atom": 5, "inversion_atom": 9}
    diEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "inversion_atom": 3}

    dimer = makeMolecule("H5O2pls", diEmbedDict, dimension="2D")
    dimer1D = makeMolecule("H5O2pls", diEmbedDict, dimension="1D")
    trimer = makeMolecule("H7O3pls", triEmbedDict, dimension="2D")
    trimer1D = makeMolecule("H7O3pls", triEmbedDict, dimension="1D")
    tetramer = makeMolecule("H9O4pls", tetEmbedDict, dimension="2D")
    tetramer1D = makeMolecule("H9O4pls", tetEmbedDict, dimension="1D")

    # modelHtest = ModelHarmonic(trimer, CC=True)
    # modelHtest.run_2D_DVR()
    # modelHtesT = ModelHarmonic(trimer, CC=False)
    # modelHtesT.run_2D_DVR()
    # modelHtest.printFreqs()
    # runTMplots(tetramer)
    runTMplots(dimer)
    # runTMplots(trimer)

    # ST = ["2D w/TDM"]
    # LT = ["C0-", "C6-", "C5-", "C8-", "C7-", "C4-", "C3-"]
    # LT1 = ["C0-", "C6-", "C5-", "C4-", "C3-"]
    # LT2 = ["C0-", "C5-", "C8-", "C7-"]
    # FS = [-30, -20, -10, 0, 10, 20, 30]
    # FS1 = [-20, -10, 0, 10, 20]
    # FS2 = [-15, -5, 5, 15]
    # TT = ["Dipole Surface", "Cubic", "Quadratic", "Quadratic OH only", "Quadratic Bilinear", "Linear", "Linear OH only"]
    # # TT1 = ["Dipole Surface", "Cubic", "Quadratic", "Linear", "Linear OH only"]
    # # TT2 = ["Dipole Surface", "Quadratic", "Quadratic OH only", "Quadratic Bilinear"]
    # fn2 = "di2DComponentSpectrum_all.png"
    # fn3 = "tri2DComponentSpectrum_all.png"
    # fn4 = "tet2DComponentSpectrum_all.png"
    # makeSpectMultiple([dimer], ST*len(TT), LT, FS, TT, fn2)
    # makeSpectMultiple([trimer], ST*len(TT), LT, FS, TT, fn3)
    # makeSpectMultiple([tetramer], ST*len(TT), LT, FS, TT, fn4)
    ST1D = ["Transition Dipole Moment"]
    LT1D = ["C1-", "C6-", "C5-", "C4-", "C3-"]
    FS1D = [-20, -10, 0, 10, 20]
    TT1D = ["Poly", "Cubic", "Quadratic", "Linear", "Constant"]
    fn1D2 = "di1DHarmonic_ComponentSpectrum_all.png"
    fn1D3 = "tri1DHarmonic_ComponentSpectrum_all.png"
    fn1D4 = "tet1DHarmonic_ComponentSpectrum_all.png"
    # makeSpectMultiple([dimer1D], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D2, anharmonics=False)
    # makeSpectMultiple([trimer1D], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D3, anharmonics=False)
    # makeSpectMultiple([tetramer1D], ST1D*len(TT1D), LT1D, FS1D, TT1D, fn1D4, anharmonics=False)

    # molObjs = [trimer1D, trimer, trimer1D, trimer1D, trimer, trimer]
    # # CH, XH/OO 2D, A OH/OO 2D, A CC OH/OO 2D, H OH/OO 2D, H CC OH/OO 2D
    LT = ["C2-", "C1-", "C3-", "C6-"]
    # a a/h h/a h
    aT = ["anharmonic", "anharm/harm", "harm/anharm", "harmonic"]
    ST = ["Transition Dipole Moment"]*len(LT)
    # molObjsT = [tetramer1D] * len(LT)
    FS = [0]*len(LT)
    TT = ["Poly"]*len(LT)
    fn = "TetAdiabatVarieties_Spectrum.png"
    # makeSpectMultiple([tetramer1D], ST, LT, FS, TT, fn, adiabatTypes=aT)
    fn1 = "TriAdiabatVarieties_Spectrum.png"
    # makeSpectMultiple([trimer1D], ST, LT, FS, TT, fn1, adiabatTypes=aT)
    fn2 = "DiAdiabatVarieties_Spectrum.png"
    # makeSpectMultiple([dimer1D], ST, LT, FS, TT, fn2, adiabatTypes=aT)
 
