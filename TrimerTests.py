"""Trimer Tests for Adiabatic Approximation Scripts"""
import os
import matplotlib.pyplot as plt
from MolecularSys import Molecule
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
# from transitionmoment import TransitionMoment
from PlotSpectrum import *

dvr_dir3 = os.path.expanduser("~/udrive/H7O3pls/DVR Results/")

ProtTri = Molecule(MoleculeName="H7O3pls",
                   atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                   method="rigid",
                   scanCoords=[(0, 1), (1, 2)],
                   embed_dict={"centralO_atom": 1,
                               "xAxis_atom": 0,
                               "xyPlane_atom": 5,
                               "outerO1": None,
                               "outerO2": None,
                               "inversion_atom": 9})

ProtTriR = Molecule(MoleculeName="H7O3pls",
                    atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                    method="relax",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "xyPlane_atom": 5,
                                "outerO1": None,
                                "outerO2": None,
                                "inversion_atom": 9})

# TriAA = AdiabaticApprox(moleculeObj=ProtTriR,
#                         DVR_desiredEnergies=4,
#                         NumPts=500)
# res3 = TriAA.run_anharOH_DVR(plotPhasedWfns=False)
# resu3 = TriAA.run_OO_DVR(OHDVRres=f"{dvr_dir3}{ProtTriR.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)
# res4 = TriAA.run_harOH_DVR(plotPhasedWfns=False)
# resu4 = TriAA.run_OO_DVR(OHDVRres=f"{dvr_dir3}{ProtTriR.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)


# TriPlot = AAplots(moleculeObj=ProtTri,
#                   OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz",
#                   OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz")
# TriPlot.ohWfn_plots(wfns2plt=2)
# TriPlot.ooWfn_plots(wfns2plt=3)
# TriPlot.make_adiabatplots()

# test = TMplots(moleculeObj=ProtTri,
#                OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz")

# tut = TMplots(moleculeObj=ProtTri,
#               OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz")
# test.DipoleSurfaces()
# test.InterpolatedDips()
# tut.TransitionMoments(ylim=(-0.75, 0.3))
# plt.show()
# test.TransitionMoments(ylim=(-0.8, 0.8))
# plt.show()

specTest = Spectrum(moleculeObj=ProtTri,
                    spectType="Cubic Harmonic")
specTest.cubicharmonic(woo=421.36, woh=2473.22, color="C0-")
speTest = Spectrum(moleculeObj=ProtTri,
                   spectType="Cubic Harmonic")
speTest.cubicharmonic(woo=411.57, woh=2397.59, color="C9-")
bleep = Spectrum(moleculeObj=ProtTriR,
                 spectType="Transition Dipole Moment",
                 OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir3}{ProtTriR.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blep = Spectrum(moleculeObj=ProtTriR,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir3}{ProtTriR.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
beep = Spectrum(moleculeObj=ProtTriR,
                spectType="Transition Dipole Moment",
                OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_HarmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir3}{ProtTriR.method}_OODVR_wharmOHDVR_energies4.npz",
                DVRmethod="Harmonic")
bep = Spectrum(moleculeObj=ProtTriR,
               spectType="Franck-Condon",
               OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_HarmOHDVR_energies4.npz",
               OODVRnpz=f"{dvr_dir3}{ProtTriR.method}_OODVR_wharmOHDVR_energies4.npz",
               DVRmethod="Harmonic")
bloop = Spectrum(moleculeObj=ProtTri,
                 spectType="Transition Dipole Moment",
                 OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blop = Spectrum(moleculeObj=ProtTri,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
boop = Spectrum(moleculeObj=ProtTri,
                spectType="Transition Dipole Moment",
                OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz",
                DVRmethod="Harmonic")
bop = Spectrum(moleculeObj=ProtTri,
               spectType="Franck-Condon",
               OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz",
               OODVRnpz=f"{dvr_dir3}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz",
               DVRmethod="Harmonic")
bop.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0)

# pob = Spectrum(moleculeObj=ProtTri,
#                spectType="2D w/TDM",
#                TwoDnpz=f"{dvr_dir3}{ProtTri.method}_2D_DVR.npz")
# pob.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)

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

