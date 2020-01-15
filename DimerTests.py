"""Dimer Tests for Adiabatic Approximation Scripts"""
import os
import matplotlib.pyplot as plt
from MolecularSys import *
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *
from transitionmoment import TransitionMoment
from PlotSpectrum import *

# two molecule objects - one for rigid scans, one for relaxed scans,
ProtDi = Molecule(MoleculeName="H5O2pls",
                  dimension="1D",
                  atom_str=["O", "O", "H", "D", "D", "D", "D"],
                  method="rigid",
                  scanCoords=[(0, 1), (1, 2)],
                  embed_dict={"centralO_atom": 1,
                              "xAxis_atom": 0,
                              "inversion_atom": 3})
ProtDiR = Molecule(MoleculeName="H5O2pls",
                   dimension="1D",
                   atom_str=["O", "O", "H", "D", "D", "D", "D"],
                   method="relax",
                   scanCoords=[(0, 1), (1, 2)],
                   embed_dict={"centralO_atom": 1,
                               "xAxis_atom": 0,
                               "inversion_atom": 3})
ProtDi2D = Molecule(MoleculeName="H5O2pls",
                    dimension="2D",
                    atom_str=["O", "O", "H", "D", "D", "D", "D"],
                    method="rigid",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "inversion_atom": 3})
ProtDi2DR = Molecule(MoleculeName="H5O2pls",
                     dimension="2D",
                     atom_str=["O", "O", "H", "D", "D", "D", "D"],
                     method="relax",
                     scanCoords=[(0, 1), (1, 2)],
                     embed_dict={"centralO_atom": 1,
                                 "xAxis_atom": 0,
                                 "inversion_atom": 3})

dvr_dir = os.path.expanduser(f"~/udrive/{ProtDi.MoleculeName}/DVR Results/")

# create AdiabaticApprox Object(s)
# DiAA = AdiabaticApprox(moleculeObj=ProtDi,
#                        DVR_desiredEnergies=4,
#                        NumPts=500)
#
# DiAAR = AdiabaticApprox(moleculeObj=ProtDiR,
#                         DVR_desiredEnergies=4,
#                         NumPts=500)

# Flavor 1 - Anharmonic OH/Adiabatic Treatment
# res = DiAA.run_anharOH_DVR(plotPhasedWfns=False)
# resu = DiAA.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtDi.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)
# resR = DiAAR.run_anharOH_DVR(plotPhasedWfns=False)
# resuR = DiAAR.run_OO_DVR(OHDVRres=f"{dvr_dir}{ProtDiR.method}_AnharmOHDVR_energies4.npz", plotPhasedWfns=False)

# Flavor 3 - 2D DVR *coord shift to OO/XH
DiAA2D = AdiabaticApprox(moleculeObj=ProtDi2D,
                         DVR_desiredEnergies=4,
                         NumPts=500)
res2dee = DiAA2D.run_2D_DVR()

# create AdiabaticApprox plots
# change moleculeObj & Anharm/Harm as necessary
DiPlot = AAplots(moleculeObj=ProtDi2DR,
                 OHDVRnpz=f"{dvr_dir}{ProtDi2DR.method}_HarmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir}{ProtDi2DR.method}_OODVR_wharmOHDVR_energies4.npz")
DiPlot.make_scan_plots()
plt.show()
# DiPlot.ohWfn_plots(wfns2plt=2)
# DiPlot.ooWfn_plots(wfns2plt=3)
# DiPlot.make_adiabatplots()

# create TranisitionMoment plots
# change moleculeObj & Anharm/Harm as necessary
# tut = TMplots(moleculeObj=ProtDi,
#               OHDVRnpz=f"{dvr_dir}{ProtDi.method}_AnharmOHDVR_energies4.npz",
#               OODVRnpz=f"{dvr_dir}{ProtDi.method}_OODVR_wanharmOHDVR_energies4.npz")
# tut.DipoleSurfaces()
# tut.InterpolatedDips()
# tut.TransitionMoments()
# plt.show()

# create Spectrum objects
# Flavor 1 Spectrum
bleep = Spectrum(moleculeObj=ProtDi,
                 spectType="Transition Dipole Moment",
                 OHDVRnpz=f"{dvr_dir}{ProtDi.method}_AnharmOHDVR_energies4.npz",
                 OODVRnpz=f"{dvr_dir}{ProtDi.method}_OODVR_wanharmOHDVR_energies4.npz",
                 DVRmethod="Anharmonic")
blep = Spectrum(moleculeObj=ProtDi,
                spectType="Franck-Condon",
                OHDVRnpz=f"{dvr_dir}{ProtDi.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir}{ProtDi.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
beep = Spectrum(moleculeObj=ProtDiR,
                spectType="Transition Dipole Moment",
                OHDVRnpz=f"{dvr_dir}{ProtDiR.method}_AnharmOHDVR_energies4.npz",
                OODVRnpz=f"{dvr_dir}{ProtDiR.method}_OODVR_wanharmOHDVR_energies4.npz",
                DVRmethod="Anharmonic")
bep = Spectrum(moleculeObj=ProtDiR,
               spectType="Franck-Condon",
               OHDVRnpz=f"{dvr_dir}{ProtDiR.method}_AnharmOHDVR_energies4.npz",
               OODVRnpz=f"{dvr_dir}{ProtDiR.method}_OODVR_wanharmOHDVR_energies4.npz",
               DVRmethod="Anharmonic")

# Flavor 3 Spectrum
peb = Spectrum(moleculeObj=ProtDi2DR,
               spectType="2D w/TDM",
               TwoDnpz=f"{dvr_dir}{ProtDiR.method}_2D_DVR.npz")
peb.make_spect(normalize=True, invert=False, line_type='C0-', freq_shift=0, savefile=False)

pebb = Spectrum(moleculeObj=ProtDi2D,
                spectType="2D w/TDM",
                TwoDnpz=f"{dvr_dir}{ProtDi.method}_2D_DVR.npz")
pebb.make_spect(normalize=True, invert=False, line_type='C9-', freq_shift=0, savefile=True)

bleep.make_spect(normalize=True, invert=False, line_type='C1-', freq_shift=0)
blep.make_spect(normalize=True, invert=False, line_type='C2-', freq_shift=10)
beep.make_spect(normalize=True, invert=False, line_type='C3-', freq_shift=0)
bep.make_spect(normalize=True, invert=False, line_type='C4-', freq_shift=10)
plt.legend()
plt.show()
