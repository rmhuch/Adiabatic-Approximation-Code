"""General Tests for Adiabatic Approximation Scripts"""

from MolecularSys import Molecule
ProtTet = Molecule(MoleculeName="H9O4pls",
                   atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                   method="rigid",
                   scanCoords=[(0, 1), (1, 2)])
# print(ProtTet.scanLogs)

# from Analysis import AdiabaticApprox
# test = AdiabaticApprox(moleculeObj=ProtTet,
#                        DVR_desiredEnergies=4,
#                        NumPts=500)
# res = test.run_harOH_DVR(plotPhasedWfns=True)
# dvr_dir = "/home/netid.washington.edu/rmhuch/udrive/H9O4pls/DVR Results/"
# # resu = test.run_OO_DVR(OHDVRres=f"{dvr_dir}OH_harmDVR_numpts500_energies4.npz")
# print(res)

from Figures import AAplots
import matplotlib.pyplot as plt
test_plot = AAplots(moleculeObj=ProtTet,
                    OHDVRnpz=
                    "/home/netid.washington.edu/rmhuch/udrive/H9O4pls/DVR Results/OH_harmDVR_numpts500_energies4.npz",
                    OODVRnpz=
                    "/home/netid.washington.edu/rmhuch/udrive/H9O4pls/DVR Results/OODVR_wharmOHDVR_numpts500_energies4.npz")
# test_plot.ohWfn_plots(wfns2plt=2)
# test_plot.ooWfn_plots(wfns2plt=3)
test_plot.make_adiabatplots()
plt.show()
