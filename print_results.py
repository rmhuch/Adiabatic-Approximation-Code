from MolecularSys import *
from new2Dcalcs import makeMolecule

tetEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "outerO1": 5, "outerO2": 8, "inversion_atom": 8} 
moleculeObj = makeMolecule("H7O3pls", tetEmbedDict, dimension="2D",OH=True)
scan_dat = np.load(os.path.join(moleculeObj.mol_dir, "DVR Results", f"{moleculeObj.method}_2D_DVR_KC_OHPot.npz"))
xh_dat = np.load(os.path.join(moleculeObj.mol_dir, "DVR Results", f"{moleculeObj.method}_2D_DVR_KC_XHPot.npz"))
harm_dat = np.load(os.path.join(moleculeObj.mol_dir, "DVR Results", "HMP_2D_DVR_KC.npz"))
ohinxh_dat = np.load(os.path.join(moleculeObj.mol_dir, "DVR Results", f"OHgridinXHpot_2D_DVR.npz"))
xhinoh_dat = np.load(os.path.join(moleculeObj.mol_dir, "DVR Results", f"XHgridinOHpot_2D_DVR.npz"))

print("OH/OO dvr", scan_dat["energy_array"])
print("XH/OO dvr", xh_dat["energy_array"])
print("OH/OO in XH/OO dvr", ohinxh_dat["energy_array"])
print("XH/OO in OH/OO dvr", xhinoh_dat["energy_array"])
print("HO OH/OO dvr", harm_dat["energy_array"])
