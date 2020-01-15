import os
import matplotlib.pyplot as plt
from MolecularSys import Molecule
from AdiabaticAnalysis import AdiabaticApprox
from Figures import *

dvr_dir4 = os.path.expanduser("~/udrive/H9O4pls/DVR Results/")
dvr_dir3 = os.path.expanduser("~/udrive/H7O3pls/DVR Results/")
dvr_dir2 = os.path.expanduser("~/udrive/H5O2pls/DVR Results/")

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

ProtTri = Molecule(MoleculeName="H7O3pls",
                   atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                   method="rigid",
                   scanCoords=[(0, 1), (1, 2)],
                   embed_dict={"centralO_atom": 1,
                               "xAxis_atom": 0,
                               "xyPlane_atom": 5,
                               "inversion_atom": 9})


ProtDi = Molecule(MoleculeName="H5O2pls",
                  atom_str=["O", "O", "H", "D", "D", "D", "D"],
                  method="rigid",
                  scanCoords=[(0, 1), (1, 2)],
                  embed_dict={"centralO_atom": 1,
                              "xAxis_atom": 0,
                              "xyPlane_atom": None})

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

TetAA = AdiabaticApprox(moleculeObj=ProtTet,
                        DVR_desiredEnergies=4,
                        NumPts=500)

res4 = TetAA.run_harOH_DVR(plotPhasedWfns=False)
resu4 = TetAA.run_OO_DVR(OHDVRres=f"{dvr_dir4}{ProtTet.method}_HarmOHDVR_energies4.npz")

ProtTriR = Molecule(MoleculeName="H7O3pls",
                    atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"],
                    method="relax",
                    scanCoords=[(0, 1), (1, 2)],
                    embed_dict={"centralO_atom": 1,
                                "xAxis_atom": 0,
                                "xyPlane_atom": 5,
                                "inversion_atom": 9})

ProtDiR = Molecule(MoleculeName="H5O2pls",
                   atom_str=["O", "O", "H", "D", "D", "D", "D"],
                   method="relax",
                   scanCoords=[(0, 1), (1, 2)],
                   embed_dict={"centralO_atom": 1,
                               "xAxis_atom": 0,
                               "xyPlane_atom": None})

# Plot = AAplots(moleculeObj=ProtTet,
#                OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_HarmOHDVR_energies4.npz",
#                OODVRnpz=f"{dvr_dir4}{ProtTet.method}_OODVR_wharmOHDVR_energies4.npz")
# Plot.ohWfn_plots(wfns2plt=2)
# Plot.ooWfn_plots(wfns2plt=3)

# make this work in case wfn txt files need to be remade.
# oos = np.array((j, 0))
# vals = np.column_stack((grid, wfns[i, :, 0]))
# data = np.vstack((oos, vals))
# np.savetxt(f"{self.molecule.method}Anharm{self.molecule.MoleculeName[:2]}_gswfn_roo{j}.txt", data)


# tes4 = AnnePlots(moleculeObj=ProtTet,
#                  OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_AnharmOHDVR_energies4.npz")
# tes3 = AnnePlots(moleculeObj=ProtTri,
#                  OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_AnharmOHDVR_energies4.npz")
# tes2 = AnnePlots(moleculeObj=ProtDi,
#                  OHDVRnpz=f"{dvr_dir2}{ProtDi.method}_AnharmOHDVR_energies4.npz")
# tes4H = AnnePlots(moleculeObj=ProtTet,
#                   OHDVRnpz=f"{dvr_dir4}{ProtTet.method}_HarmOHDVR_energies4.npz")
# tes3H = AnnePlots(moleculeObj=ProtTri,
#                   OHDVRnpz=f"{dvr_dir3}{ProtTri.method}_HarmOHDVR_energies4.npz")
# tes2H = AnnePlots(moleculeObj=ProtDi,
#                   OHDVRnpz=f"{dvr_dir2}{ProtDi.method}_HarmOHDVR_energies4.npz")
# tes4RH = AnnePlots(moleculeObj=ProtTetR,
#                    OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_HarmOHDVR_energies4.npz")
# tes3RH = AnnePlots(moleculeObj=ProtTriR,
#                    OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_HarmOHDVR_energies4.npz")
# tes2RH = AnnePlots(moleculeObj=ProtDiR,
#                    OHDVRnpz=f"{dvr_dir2}{ProtDiR.method}_HarmOHDVR_energies4.npz")
# tes4R = AnnePlots(moleculeObj=ProtTetR,
#                   OHDVRnpz=f"{dvr_dir4}{ProtTetR.method}_AnharmOHDVR_energies4.npz")
# tes3R = AnnePlots(moleculeObj=ProtTriR,
#                   OHDVRnpz=f"{dvr_dir3}{ProtTriR.method}_AnharmOHDVR_energies4.npz")
# tes2R = AnnePlots(moleculeObj=ProtDiR,
#                   OHDVRnpz=f"{dvr_dir2}{ProtDiR.method}_AnharmOHDVR_energies4.npz")

# tes4.eqOHPlot(color="red")
# tes4R.eqOHPlot(color="maroon")
# tes3.eqOHPlot(color="green")
# tes3R.eqOHPlot(color="darkolivegreen")
# # tes2.eqOHPlot(color="purple")
# # tes2R.eqOHPlot(color="indigo")
# plt.legend(fontsize="small")
# plt.show()

# tes4.freqOHPlot(color="red")
# # tes4H.freqOHPlot(color="orangered")
# tes4R.freqOHPlot(color="maroon")
# # tes4RH.freqOHPlot(color="firebrick")
#
# tes3.freqOHPlot(color="green")
# # tes3H.freqOHPlot(color="springgreen")
# tes3R.freqOHPlot(color="darkolivegreen")
# # tes3RH.freqOHPlot(color="olivedrab")
# #
# # # tes2.freqOHPlot(color="purple")
# # tes2H.freqOHPlot(color="darkviolet")
# # # tes2R.freqOHPlot(color="indigo")
# # tes2RH.freqOHPlot(color="rebeccapurple")
# plt.legend(fontsize="small")
# plt.show()

