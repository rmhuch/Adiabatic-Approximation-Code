"""General Tests for Adiabatic Approximation Scripts"""

from MolecularSys import Molecule
ProtTet = Molecule(MoleculeName="H9O4pls",
                   atom_str=["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"],
                   method="relax",
                   scanCoords=[(0, 1), (1, 2)])
# print(ProtTet.scanLogs)
from Analysis import AdiabaticApprox
test = AdiabaticApprox(moleculeObj=ProtTet,
                       DVR_desiredEnergies=4,
                       NumPts=500)
res = test.run_anharOH_DVR()
print(res[1])


