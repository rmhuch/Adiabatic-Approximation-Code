import matplotlib.pyplot as plt
from MolecularSys import *
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
from transitionmoment import TransitionMoment
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
            ohdvr_filename = f"{dvr_dir}{molObj.method}_AnharmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}{molObj.method}_OODVR_wanharmOHDVR_energies{numDVRenergies}.npz"
        else:
            ohdvr_filename = f"{dvr_dir}{molObj.method}_HarmOHDVR_energies{numDVRenergies}.npz"
            oodvr_filename = f"{dvr_dir}{molObj.method}_OODVR_wharmOHDVR_energies{numDVRenergies}.npz"
        tut = TMplots(moleculeObj=molObj, OHDVRnpz=ohdvr_filename, OODVRnpz=oodvr_filename)
        tut.DipoleSurfaces()
        tut.TransitionMoments(ylim=(-0.4, 0.8))
        tut.InterpolatedDips()
        tut.componentTMs()
    else:
        tot = TM2Dplots(moleculeObj=molObj,
                        TwoDnpz=f"{dvr_dir}{molObj.method}_2D_DVR.npz")
        tot.DipoleSurfaces()
        tot.componentTMs()


if __name__ == '__main__':
    triEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "xyPlane_atom": 5, "inversion_atom": 9}
    diEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "inversion_atom": 3}
    test = makeMolecule("H7O3pls", triEmbedDict)
    print(test.logData.cartesians)
