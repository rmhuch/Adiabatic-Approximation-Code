"""Tetramer Tests for Adiabatic Approximation Scripts"""
import os
import matplotlib.pyplot as plt
from MolecularSys import *
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
from transitionmoment import TransitionMoment
from PlotSpectrum import *

# two molecule objects - one for rigid scans, one for relaxed scans,
ProtTet = Molecule(MoleculeName="H9O4pls",
                   dimension="1D",
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
                    dimension="1D",
                    atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                    method="relax",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "xyPlane_atom": None,
                                "outerO1": 5,
                                "outerO2": 8,
                                "inversion_atom": 8})
ProtTet2D = Molecule(MoleculeName="H9O4pls",
                     dimension="2D",
                     atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                     method="rigid",
                     scanCoords=[(0, 1), (1, 2)],
                     embed_dict={"centralO_atom": 1,
                                 "xAxis_atom": 0,
                                 "xyPlane_atom": None,
                                 "outerO1": 5,
                                 "outerO2": 8,
                                 "inversion_atom": 8})
ProtTet2DR = Molecule(MoleculeName="H9O4pls",
                      dimension="2D",
                      atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                      method="relax",
                      scanCoords=[(0, 1), (1, 2)],
                      embed_dict={"centralO_atom": 1,
                                  "xAxis_atom": 0,
                                  "xyPlane_atom": None,
                                  "outerO1": 5,
                                  "outerO2": 8,
                                  "inversion_atom": 8})

dvr_dir = os.path.expanduser(f"~/udrive/{ProtTet.MoleculeName}/DVR Results/")

# create AdiabaticApprox Object(s)
# TetAA = AdiabaticApprox(moleculeObj=ProtTet,
#                         DVR_desiredEnergies=4,
#                         NumPts=500)
#
# TetAAR = AdiabaticApprox(moleculeObj=ProtTetR,
#                          DVR_desiredEnergies=4,
#                          NumPts=500)

# Flavor 1 - Anharmonic OH/Adiabatic Treatment
# res = TetAA.run_anharOH_DVR(plotPhasedWfns=False)
# resu = TetAA.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)
# resR = TetAAR.run_anharOH_DVR(plotPhasedWfns=False)
# resuR = TetAAR.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTetR.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)

# Flavor 2 - Harmonic OH/Adiabatic Treatment
# resOH = TetAA.run_harOH_DVR(plotPhasedWfns=False)
# resuOH = TetAA.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTet.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)
#
# resROH = TetAAR.run_harOH_DVR(plotPhasedWfns=False)
# resuROH = TetAAR.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTetR.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)

# Flavor 3 - 2D DVR *coord shift to OO/XH
# TetAA2D = AdiabaticApprox(moleculeObj=ProtTet2D,
#                           DVR_desiredEnergies=4,
#                           NumPts=500)
# res2dee = TetAA2D.run_2D_DVR()

# create AdiabaticApprox plots
# change moleculeObj & Anharm/Harm as necessary
TetPlot = AAplots(moleculeObj=ProtTet2DR,
                  OHDVRnpz=f"{dvr_dir}{ProtTet2DR.method}_AnharmOHDVR_energies4.npz",
                  OODVRnpz=f"{dvr_dir}{ProtTet2DR.method}_OODVR_wanharmOHDVR_energies4.npz")
TetPlot.make_scan_plots()
plt.show()
# TetPlot.ohWfn_plots(wfns2plt=2)
# TetPlot.ooWfn_plots(wfns2plt=3)
# TetPlot.make_adiabatplots()

# create TranisitionMoment plots
# change moleculeObj & Anharm/Harm as necessary
# tut = TMplots(moleculeObj=ProtTet,
#               OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz")
# tut.DipoleSurfaces()
# tut.InterpolatedDips()
# tut.TransitionMoments(ylim=(-0.8, 0.8))
# plt.show()

# create Spectrum objects
# Flavor 1 Spectrum
# bleep = Spectrum(moleculeObj=ProtTet,
#                  spectType="Transition Dipole Moment",
#                  OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# blep = Spectrum(moleculeObj=ProtTet,
#                 spectType="Franck-Condon",
#                 OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# beep = Spectrum(moleculeObj=ProtTetR,
#                 spectType="Transition Dipole Moment",
#                 OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# bep = Spectrum(moleculeObj=ProtTetR,
#                spectType="Franck-Condon",
#                OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                DVRmethod="Anharmonic")
# # Flavor 2 Spectrum
# bloop = Spectrum(moleculeObj=ProtTet,
#                  spectType="Transition Dipole Moment",
#                  OHDVRnpz=f"{dvr_dir}{ProtTet.method}_HarmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wharmOHDVR_energies4.npz",
#                  DVRmethod="Harmonic")
# blop = Spectrum(moleculeObj=ProtTet,
#                 spectType="Franck-Condon",
#                 OHDVRnpz=f"{dvr_dir}{ProtTet.method}_HarmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wharmOHDVR_energies4.npz",
#                 DVRmethod="Harmonic")
# boop = Spectrum(moleculeObj=ProtTetR,
#                 spectType="Transition Dipole Moment",
#                 OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_HarmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wharmOHDVR_energies4.npz",
#                 DVRmethod="Harmonic")
# bop = Spectrum(moleculeObj=ProtTetR,
#                spectType="Franck-Condon",
#                OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_HarmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wharmOHDVR_energies4.npz",
#                DVRmethod="Harmonic")
# Flavor 3 Spectrum
# peb = Spectrum(moleculeObj=ProtTet2DR,
#                spectType="2D w/TDM",
#                TwoDnpz=f"{dvr_dir}{ProtTetR.method}_2D_DVR.npz")
# peb.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0, savefile=True)
#
# pebb = Spectrum(moleculeObj=ProtTet2D,
#                 spectType="2D w/TDM",
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebb.make_spect(normalize=True, invert=False, line_type='C9-', freq_shift=0, savefile=True)
# Flavor 4 Spectrum
# pobb = Spectrum(moleculeObj=ProtTetR,
#                 spectType="Cubic Harmonic")
# pobb.cubicharmonic(woo=362.58, woh=2894.68, fancyF=102, color='C9-')

# bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)
# blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=10)
# beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
# bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
# bloop.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
# blop.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=10)
# boop.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=0)
# bop.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=10)
# plt.legend()
# plt.show()
