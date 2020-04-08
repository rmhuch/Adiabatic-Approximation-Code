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
#                         DVR_desiredEnergies=1,
#                         NumPts=2000)
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
TetAA2D = AdiabaticApprox(moleculeObj=ProtTet2D,
                          DVR_desiredEnergies=4,
                          NumPts=500)
res2dee = TetAA2D.run_2D_DVR()

# create AdiabaticApprox plots
# change moleculeObj & Anharm/Harm as necessary
# TetPlot = AAplots(moleculeObj=ProtTet2D,
#                   OHDVRnpz=f"{dvr_dir}{ProtTet.method}_anharmOHDVR_energies4.npz",
#                   OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz")
# TetPlot.make_scan_plots(grid=True)
# plt.show()
# TetPlot.ohWfn_plots(wfns2plt=2)
# TetPlot.ooWfn_plots(wfns2plt=3)
# TetPlot.make_adiabatplots()

# create TranisitionMoment plots
# change moleculeObj & Anharm/Harm as necessary
# tut = TMplots(moleculeObj=ProtTet,
#               OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz")
# tut.DipoleSurfaces()
# tut.TransitionMoments(ylim=(-0.4, 0.8))
# tut.InterpolatedDips()
# tut.componentTMs()
# plt.show()

# create 2D TDM plots
# tot = TM2Dplots(moleculeObj=ProtTet,
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# tot.DipoleSurfaces()
# tot.componentTMs()

# create Spectrum objects
# Flavor 1 Spectrum
# bleep = Spectrum(moleculeObj=ProtTet,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Poly",
#                  OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=-20)
# blep = Spectrum(moleculeObj=ProtTet,
#                 spectType="Franck-Condon",
#                 OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=50)
# beep = Spectrum(moleculeObj=ProtTetR,
#                 spectType="Transition Dipole Moment",
#                 OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
# bep = Spectrum(moleculeObj=ProtTetR,
#                spectType="Franck-Condon",
#                OHDVRnpz=f"{dvr_dir}{ProtTetR.method}_AnharmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir}{ProtTetR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                DVRmethod="Anharmonic")
# bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
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
# bloop.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
# blop.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=0)
# boop.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=0)
# bop.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=10)
# Flavor 3 Spectrum
# pebb = Spectrum(moleculeObj=ProtTet2D,
#                 spectType="2D w/TDM",
#                 TDMtype="Poly",
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebb.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0, savefile=True)
# Flavor 4 Spectrum
# pobb = Spectrum(moleculeObj=ProtTet,
#                 spectType="Cubic Harmonic")
# pobb.cubicharmonic(woo=362.58, woh=2894.68, fancyF=102, color='C9-')

# Flavor 1 Spectrum Components
# tesst = Spectrum(moleculeObj=ProtTet,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Linear",
#                  OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# test = Spectrum(moleculeObj=ProtTet,
#                 spectType="Transition Dipole Moment",
#                 TDMtype="Constant",
#                 OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# tessst = Spectrum(moleculeObj=ProtTet,
#                   spectType="Transition Dipole Moment",
#                   TDMtype="Quadratic",
#                   OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                   OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                   DVRmethod="Anharmonic")
# tesssst = Spectrum(moleculeObj=ProtTet,
#                    spectType="Transition Dipole Moment",
#                    TDMtype="Cubic",
#                    OHDVRnpz=f"{dvr_dir}{ProtTet.method}_AnharmOHDVR_energies4.npz",
#                    OODVRnpz=f"{dvr_dir}{ProtTet.method}_OODVR_wanharmOHDVR_energies4.npz",
#                    DVRmethod="Anharmonic")
# test.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=20)
# tesst.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
# tessst.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
# tesssst.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=-10)

# Flavor 3 Spectrum Components
# pebDS = Spectrum(moleculeObj=ProtTet2D,
#                  spectType="2D w/TDM",
#                  TDMtype="Dipole Surface",
#                  TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebC = Spectrum(moleculeObj=ProtTet2D,
#                 spectType="2D w/TDM",
#                 TDMtype="Cubic",
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebQ = Spectrum(moleculeObj=ProtTet2D,
#                 spectType="2D w/TDM",
#                 TDMtype="Quadratic",
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebBQ = Spectrum(moleculeObj=ProtTet2D,
#                  spectType="2D w/TDM",
#                  TDMtype="Quadratic Bilinear",
#                  TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebOHQ = Spectrum(moleculeObj=ProtTet2D,
#                   spectType="2D w/TDM",
#                   TDMtype="Quadratic OH only",
#                   TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebL = Spectrum(moleculeObj=ProtTet2D,
#                 spectType="2D w/TDM",
#                 TDMtype="Linear",
#                 TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebCo = Spectrum(moleculeObj=ProtTet2D,
#                  spectType="2D w/TDM",
#                  TDMtype="Linear OH only",
#                  TwoDnpz=f"{dvr_dir}{ProtTet.method}_2D_DVR.npz")
# pebDS.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=-30)
# pebC.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=-20)
# pebQ.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=-10)
# pebOHQ.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=0)
# pebBQ.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=10)
# pebL.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=20)
# pebCo.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=30)
# plt.legend(fontsize=12)
# plt.show()
