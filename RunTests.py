from MolecularSys import *
from AdiabaticAnalysis import *
from Figures import *
from PlotSpectrum import *
from PotExpansions import *
from transitionmoment import *

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
            # AAobj.run_anharOH_DVR(plotPhasedWfns=plotPhasedWfns)
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
            # PlotObj.ohWfn_PAs()
            # PlotObj.make_scan_plots()
            # PlotObj.ohWfn_plots(wfns2plt=ohwfns2plt)
            # PlotObj.ooWfn_plots(wfns2plt=oowfns2plt)
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
        tut = TMplots(moleculeObj=molObj, TwoDnpz=f"{dvr_dir}/{molObj.method}_2D_DVR.npz",
                      OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
        # tet.psi_trans()
        # tut.DipoleSurfaces()
        # tut.InterpolatedDips()
        # di = [2.163, 2.403, 2.583], tri = [2.3266, 2.5066, 2.7466], tet = [2.3296, 2.5696, 2.8096]
        # x = di (2.1, 2.6) tri (2.3, 2.8) tet (2.3, 2.9)
        # y = di (-1.6, -0.6) tri (-0.85, -0.45) tet (0.3, 0.75)
        tut.componentTMs([2.163, 2.403, 2.583], xlim=(2.1, 2.6), ylim=(-1.6, -0.6))  # xlim OO
    else:
        tot = TM2Dplots(moleculeObj=molObj,
                        TwoDnpz=f"{dvr_dir}/{molObj.method}_2D_DVR.npz")
        # tet = TransitionMoment(moleculeObj=molObj, dimension="2D", TwoDnpz=f"{dvr_dir}/{molObj.method}_2D_DVR.npz")
        # tet.calc_all2Dmus()
        # tot.DipoleSurfaces()
        # tot.componentDMs()
        tot.plotDMcut(ylim=(-6, 6), xlim=(-0.3, 0.6))  # xlim XH

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
            twoDnpz = f"{dvr_dir}/HMP_2D_DVR_XHOO.npz"
    elif molObj.dimension == "1D":
        twoDnpz = f"{dvr_dir}/{molObj.method}_2D_DVR.npz"
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
    print(SpectValues["title"], SpectValues["frequencies"], SpectValues["norm_intensities"], SpectValues["intensities"])
    return SpectValues, SpectFig

def makeSpectMultiple(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, filename,
                      adiabatTypes=None, inverts=False, normalize=True, numDVRenergies=4, graph=True):
    if len(molObjs) == 1:
        molObjs = molObjs*len(spectTypes)
    if isinstance(inverts, bool):
        inverts = [inverts]*len(spectTypes)
    if adiabatTypes is None:
        adiabatTypes = [None]*len(spectTypes)
    SpectFig = None
    AllSpectValues = []
    for molObj, spectType, lineType, freq_shift, TDMtype, adiabatType, invert in \
            zip(molObjs, spectTypes, lineTypes, freq_shifts, TDMtypes, adiabatTypes, inverts):
        if spectType == "Cubic Harmonic":
            if molObj.MoleculeName == "H9O4pls":
                CHobj = CubicHarmonic(molObj, omegaOO=332, omegaOH=2959, FancyF=299) #  omegaOO=362.32, omegaOH=2909.44)
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
        if graph:
            AllSpectValues.append(SpectValues)
    if filename is not None:
        SpectFig[0].savefig(filename)
        plt.close()
    else:
        # SpectFig[0].show()
        plt.close()
    if graph:
        makeCompPlot(AllSpectValues)
    return AllSpectValues

def makeCompPlot(allSpectValues, scatter=True):
    import matplotlib.pyplot as plt
    # 2D DM full, Ad TDM full, HMP DM full, 2D DM quad, Ad TDM quad, HMP DM quad, 2D DM linOH, AdTDM linOH, HMP DM linOH
    y = np.zeros(len(allSpectValues))
    for i in range(len(allSpectValues)):
        vals = allSpectValues[i]["norm_intensities"]
        y[i] = (vals[1]/vals[0])*100

    barWidth = 0.15
    full = y[:3]
    quad = y[3:6]
    lin = y[6:]
    r1 = np.arange(len(full))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    if scatter:
        plt.scatter(r2, full, color="violet", label="Full Dipole Expression")
        plt.scatter(r2, quad, color="darkviolet", label="Quadratic Dipole Expression")
        plt.scatter(r2, lin, color="darkblue", label="Linear Dipole Expression")
    else:
        plt.bar(r1, full, color="violet", width=barWidth, edgecolor='white', label="Full Dipole Expression")
        plt.bar(r2, quad, color="darkviolet", width=barWidth, edgecolor='white', label="Quadratic Dipole Expression")
        plt.bar(r3, lin, color="darkblue", width=barWidth, edgecolor='white', label="Linear Dipole Expression")

    plt.xticks([r + barWidth for r in range(len(full))], ['2D Potential', 'Adiabatic Potential',
                                                          'Harmonic Coupling'])
    plt.ylabel(r"$\dfrac{I_{1, 0}}{I_{0, 0}}$", rotation=0, fontweight='bold', fontsize=16)
    plt.xlabel('Potential Energy Coupling', fontweight='bold', fontsize=12)
    plt.legend()
    plt.show()

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
    runTMplots(dimer)
    runTMplots(trimer)
    runTMplots(tetramer)
    # runAdiabaticApprox(tetramer1D)
    # runAdiabaticApprox(trimer1D)
    # runAdiabaticApprox(dimer1D)
    # # Calculate the TDM using new expansion.
    # ST1D = ["Transition Dipole Moment"]
    # LT1D = ["C1-", "C6-", "C5-", "C4-", "C3-", "C0-", "C2-"]
    # FS1D = [-30, -20, -10, 0, 10, 20, 30]
    # TT1D = ["Dipole Surface", "Cubic", "Quadratic", "Quadratic OH only", "Quadratic Bilinear", "Linear", "Linear OH only"]
    # aT = ["anharmonic"]*len(TT1D)
    # fn1D4 = "tri1DComponentSpectrum_all.png"
    # makeSpectMultiple([dimer1D], ST1D * len(TT1D), LT1D, FS1D, TT1D, fn1D4, adiabatTypes=aT, graph=False)

    # molObjs = [tetramer, tetramer1D, tetramer]*3
    # molObjsT = [trimer, trimer1D, trimer]*3
    # molObjsD = [dimer, dimer1D]*3
    # CH, XH/OO 2D, A OH/OO 2D, A CC OH/OO 2D, H OH/OO 2D, H CC OH/OO 2D
    # LT = ["C0-", "C1-", "C3-"]*3
    # LTd = ["C0-", "C1-"]*3
    # aT = [None, "anharmonic", None]*3
    # aTd = [None, "anharmonic"]*3
    # 2D DM full, Ad TDM full, HMP DM full, 2D DM quad, Ad TDM quad, HMP DM quad, 2D DM lin, Ad TDM lin, HMP DM lin
    # STm = ["2D w/TDM", "Transition Dipole Moment", "Harmonic Model"]*3
    # STmd = ["2D w/TDM", "Transition Dipole Moment"] * 3
    # FS = [0]*len(STm)
    # FSd = [0] * len(STmd)
    # TT = ["Dipole Surface", "Dipole Surface", "Dipole Surface", "Quadratic OH only", "Quadratic OH only", "Quadratic OH only",
    #       "Linear OH only", "Linear OH only", "Linear OH only"]
    # TTd = ["Dipole Surface", "Dipole Surface", "Quadratic OH only", "Quadratic OH only",
    #        "Linear OH only", "Linear OH only"]
    # makeSpectMultiple(molObjsT, STm, LT, FS, TT, filename=None, adiabatTypes=aT, graph=False)
    # makeSpectMultiple(molObjsD, STmd, LTd, FSd, TTd, filename=None, adiabatTypes=aTd, graph=False)
    # CHobj = CubicHarmonic(trimer, omegaOO=347, omegaOH=1896, FancyF=285)
    makeSpectSingle(trimer, "Harmonic Model", "C0-", TDMtype="Dipole Surface", adiabatType="anharmonic")
    makeSpectSingle(trimer, "Harmonic Model", "C0-", TDMtype="Quadratic OH only", adiabatType="anharmonic")
    makeSpectSingle(trimer, "Harmonic Model", "C0-", TDMtype="Linear OH only", adiabatType="anharmonic")

