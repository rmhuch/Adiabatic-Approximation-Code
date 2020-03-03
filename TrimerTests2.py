"""Trimer Tests for Adiabatic Approximation Scripts"""
import os
import matplotlib.pyplot as plt
from MolecularSys import *
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
from transitionmoment import TransitionMoment
from PlotSpectrum import *

# two molecule objects - one for rigid scans, one for relaxed scans,
ProtTri = Molecule(MoleculeName="H7O3pls",
                   dimension="1D",
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
                    dimension="1D",
                    atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                    method="relax",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "xyPlane_atom": 5,
                                "outerO1": None,
                                "outerO2": None,
                                "inversion_atom": 9})
ProtTri2D = Molecule(MoleculeName="H7O3pls",
                     dimension="2D",
                     atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                     method="rigid",
                     scanCoords=[(0, 1), (1, 2)],
                     embed_dict={"centralO_atom": 1,
                                 "xAxis_atom": 0,
                                 "xyPlane_atom": 5,
                                 "outerO1": None,
                                 "outerO2": None,
                                 "inversion_atom": 9})
ProtTri2DR = Molecule(MoleculeName="H7O3pls",
                      dimension="2D",
                      atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                      method="relax",
                      scanCoords=[(0, 1), (1, 2)],
                      embed_dict={"centralO_atom": 1,
                                  "xAxis_atom": 0,
                                  "xyPlane_atom": 5,
                                  "outerO1": None,
                                  "outerO2": None,
                                  "inversion_atom": 9})

dvr_dir = os.path.expanduser(f"~/udrive/{ProtTri.MoleculeName}/DVR Results/")

# create AdiabaticApprox Object(s)
TriAA = AdiabaticApprox(moleculeObj=ProtTri,
                        DVR_desiredEnergies=1,
                        NumPts=2000)
#
# TriAAR = AdiabaticApprox(moleculeObj=ProtTriR,
#                          DVR_desiredEnergies=4,
#                          NumPts=500)

# Flavor 1 - Anharmonic OH/Adiabatic Treatment
res = TriAA.run_anharOH_DVR(plotPhasedWfns=False)
# resu = TriAA.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)
# resR = TriAAR.run_anharOH_DVR(plotPhasedWfns=False)
# resuR = TriAAR.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTriR.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)

# Flavor 2 - Harmonic OH/Adiabatic Treatment
# resOH = TriAA.run_harOH_DVR(plotPhasedWfns=False)
# resuOH = TriAA.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTri.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)
#
# resROH = TriAAR.run_harOH_DVR(plotPhasedWfns=False)
# resuROH = TriAAR.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtTriR.method}_HarmOHDVR_energies4.npz", plotPhasedWfns=False)

# Flavor 3 - 2D DVR *coord shift to OO/XH
# TriAA2D = AdiabaticApprox(moleculeObj=ProtTri2DR,
#                           DVR_desiredEnergies=4,
#                           NumPts=500)
# res2dee = TriAA2D.run_2D_DVR()

# create AdiabaticApprox plots
# change moleculeObj & Anharm/Harm as necessary
# TriPlot = AAplots(moleculeObj=ProtTri,
#                   OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#                   OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz")
# TriPlot.make_scan_plots()
# plt.show()
# TriPlot.ohWfn_plots(wfns2plt=2)
# TriPlot.ooWfn_plots(wfns2plt=3)
# TriPlot.make_adiabatplots()

# create frequency plots
# righarm = AnnePlots(moleculeObj=ProtTri,
#                     OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz")
# riganharm = AnnePlots(moleculeObj=ProtTri,
#                       OHDVRnpz=f"{dvr_dir}{ProtTri.method}_harmOHDVR_energies4.npz")
# righarm.freqOHPlot(color="r")
# riganharm.freqOHPlot(color="b")
# plt.legend()
# plt.show()

# create TranisitionMoment plots
# change moleculeObj & Anharm/Harm as necessary
# tut = TMplots(moleculeObj=ProtTri,
#               OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz")
# tut.DipoleSurfaces()
# tut.InterpolatedDips()
# tut.TransitionMoments(ylim=(-0.8, 0.8))
# plt.show()

# create Spectrum objects
# Flavor 1 Spectrum
bleep = Spectrum(moleculeObj=ProtTri,
                 spectType="Transition Dipole Moment",
                 TDMtype="Full",
                 OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blep = Spectrum(moleculeObj=ProtTri,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
# beep = Spectrum(moleculeObj=ProtTriR,
#                 spectType="Transition Dipole Moment",
#                 OHDVRnpz=f"{dvr_dir}{ProtTriR.method}_AnharmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTriR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                 DVRmethod="Anharmonic")
# bep = Spectrum(moleculeObj=ProtTriR,
#                spectType="Franck-Condon",
#                OHDVRnpz=f"{dvr_dir}{ProtTriR.method}_AnharmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir}{ProtTriR.method}_OODVR_wanharmOHDVR_energies4.npz",
#                DVRmethod="Anharmonic")
#
# # Flavor 2 Spectrum
# bloop = Spectrum(moleculeObj=ProtTri,
#                  spectType="Transition Dipole Moment",
#                  OHDVRnpz=f"{dvr_dir}{ProtTri.method}_HarmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz",
#                  DVRmethod="Harmonic")
# blop = Spectrum(moleculeObj=ProtTri,
#                 spectType="Franck-Condon",
#                 OHDVRnpz=f"{dvr_dir}{ProtTri.method}_HarmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wharmOHDVR_energies4.npz",
#                 DVRmethod="Harmonic")
# boop = Spectrum(moleculeObj=ProtTriR,
#                 spectType="Transition Dipole Moment",
#                 OHDVRnpz=f"{dvr_dir}{ProtTriR.method}_HarmOHDVR_energies4.npz",
#                 OODVRnpz=f"{dvr_dir}{ProtTriR.method}_OODVR_wharmOHDVR_energies4.npz",
#                 DVRmethod="Harmonic")
# bop = Spectrum(moleculeObj=ProtTriR,
#                spectType="Franck-Condon",
#                OHDVRnpz=f"{dvr_dir}{ProtTriR.method}_HarmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir}{ProtTriR.method}_OODVR_wharmOHDVR_energies4.npz",
#                DVRmethod="Harmonic")
# Flavor 3 Spectrum
# peb = Spectrum(moleculeObj=ProtTri2DR,
#                spectType="2D w/TDM",
#                TwoDnpz=f"{dvr_dir}{ProtTriR.method}_2D_DVR.npz")
# peb.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0, savefile=False)

pebb = Spectrum(moleculeObj=ProtTri2D,
                spectType="2D w/TDM",
                TwoDnpz=f"{dvr_dir}{ProtTri.method}_2D_DVR.npz")
pebb.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0, savefile=False)
# Flavor 4 Spectrum
pobb = Spectrum(moleculeObj=ProtTri,
                spectType="Cubic Harmonic")
pobb.cubicharmonic(woo=421.37, woh=2508.70, fancyF=102, color='C9-')

bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)
blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=10)
# beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
# bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
# bloop.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
# blop.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=10)
# boop.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=0)
# bop.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=10)
# plt.legend(fontsize=16)
# plt.show()

# tesst = Spectrum(moleculeObj=ProtTri,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Linear",
#                  OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# test = Spectrum(moleculeObj=ProtTri,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Constant",
#                  OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# tessst = Spectrum(moleculeObj=ProtTri,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Quadratic",
#                  OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# tesssst = Spectrum(moleculeObj=ProtTri,
#                  spectType="Transition Dipole Moment",
#                  TDMtype="Cubic",
#                  OHDVRnpz=f"{dvr_dir}{ProtTri.method}_AnharmOHDVR_energies4.npz",
#                  OODVRnpz=f"{dvr_dir}{ProtTri.method}_OODVR_wanharmOHDVR_energies4.npz",
#                  DVRmethod="Anharmonic")
# bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)
# blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=50)
# test.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=40)
# tesst.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=30)
# tessst.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=20)
# tesssst.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=10)

# beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
# bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
# bloop.make_spect(normalize=True, invert=False, line_type='C5-', freq_shift=0)
# blop.make_spect(normalize=True, invert=False, line_type='C6-', freq_shift=0)
# boop.make_spect(normalize=True, invert=False, line_type='C7-', freq_shift=0)
# bop.make_spect(normalize=True, invert=False, line_type='C8-', freq_shift=10)
plt.legend(fontsize=12)
plt.show()
