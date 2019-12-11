"""General Tests for Adiabatic Approximation Scripts"""
import os
import matplotlib.pyplot as plt
from MolecularSys import Molecule
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
from transitionmoment import TransitionMoment
from PlotSpectrum import *

dvr_dir4 = os.path.expanduser("~/udrive/H9O4pls/DVR Results/")
dvr_dir3 = os.path.expanduser("~/udrive/H7O3pls/DVR Results/")
# dvr_dir2 = os.path.expanduser("~/udrive/H5O2pls/DVR Results/")

ProtTet = Molecule(MoleculeName="H9O4pls",
                   atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                   method="rigid",
                   scanCoords=[(0, 1), (1, 2)],
                   embed_dict={"centralO_atom": 1,
                               "xAxis_atom": 0,
                               "xyPlane_atom": None,
                               "outerO1": 5,
                               "outerO2": 8,
                               "inversion_atom": 8})

ProtTetR = Molecule(MoleculeName="H9O4pls",
                    atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                    method="relax",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "xyPlane_atom": None,
                                "outerO1": 5,
                                "outerO2": 8,
                                "inversion_atom": 8})
# print(ProtTetR.scanGrid)

# TetAA = AdiabaticApprox(moleculeObj=ProtTetR,
#                         DVR_desiredEnergies=4,
#                         NumPts=500)
# res4 = TetAA.run_harOH_DVR(plotPhasedWfns=False)
# plt.show()
# resu4 = TetAA.run_OO_DVR(OHDVRres=f"{dvr_dir4}{ProtTetR.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)
# plt.show()
#
# TetPlot = AAplots(moleculeObj=ProtTet,
#                   OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                   OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz")
# TetPlot.ohWfn_plots(wfns2plt=2)
# TetPlot.ooWfn_plots(wfns2plt=3)

# test = TMplots(moleculeObj=ProtTet,
#                OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz")

# tut = TMplots(moleculeObj=ProtTetR,
#               OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_HarmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wharmOHDVR_energies4.npz")
# tut.DipoleSurfaces()
# tut.InterpolatedDips()
# tut.TransitionMoments(ylim=(-0.75, 0.3))
# plt.show()
# test.TransitionMoments(ylim=(-0.3, 0.75))
# plt.show()
# blop = TransitionMoment(moleculeObj=ProtTetR,
#                         OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
#                         OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz")
# print(blop.embeddedDips)

bleep = Spectrum(moleculeObj=ProtTetR,
                 spectType="Transition Dipole Moment",
                 OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blep = Spectrum(moleculeObj=ProtTetR,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
beep = Spectrum(moleculeObj=ProtTetR,
                spectType="Transition Dipole Moment",
                OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_HarmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wharmOHDVR_energies4.npz",
                DVRmethod="Harmonic")
bep = Spectrum(moleculeObj=ProtTetR,
               spectType="Franck-Condon",
               OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_HarmOHDVR_energies4.npz",
               OODVRnpz=f"{dvr_dir4}{ProtTetR.method}_OODVR_wharmOHDVR_energies4.npz",
               DVRmethod="Harmonic")
bloop = Spectrum(moleculeObj=ProtTet,
                 spectType="Transition Dipole Moment",
                 OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blop = Spectrum(moleculeObj=ProtTet,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
boop = Spectrum(moleculeObj=ProtTet,
                spectType="Transition Dipole Moment",
                OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_HarmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wharmOHDVR_energies4.npz",
                DVRmethod="Harmonic")
bop = Spectrum(moleculeObj=ProtTet,
               spectType="Franck-Condon",
               OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_HarmOHDVR_energies4.npz",
               OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wharmOHDVR_energies4.npz",
               DVRmethod="Harmonic")


def osharmonic(woo, woh, fancyF):
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
    print(f"OverSimplified Spectrum Values")
    print(f"Frequencies: {energies} \n")
    print(f"Intensity: {np.sum(intensities)} \n")
    print(f"Normalized Intensities: {norm_intents} \n")
    plt.rcParams.update({'font.size': 16})
    markerline, stemline, baseline = plt.stem(energies, norm_intents,
                                              linefmt='C0-', markerfmt=' ', use_line_collection=True,
                                              label=f"Oversimplified Harmonic")
    plt.setp(stemline, 'linewidth', 6.0)
    plt.setp(baseline, visible=False)
    plt.ylim(0, 1)


osharmonic(332, 2959, 299)
bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)
blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=10)
beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
bloop.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
blop.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=10)
boop.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=0)
bop.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=10)
plt.legend()
plt.show()

