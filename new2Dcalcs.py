import numpy as np

from MolecularSys import *
from AdiabaticAnalysis import *
from Figures import *
from PlotSpectrum import *
from PotExpansions import *
from transitionmoment import *

def makeMolecule(MolDirName, embedDict, scancoords=((0, 1), (1, 2)), method="rigid", dimension="2D", OH=None):
    if MolDirName == "H9O4pls":
        atomStr = ["O", "O", "H", "D", "D", "O", "D", "D", "O", "D", "D", "D", "D"]
    elif MolDirName == "H7O3pls":
        atomStr = ["O", "O", "H", "D", "D", "O", "D", "D", "D", "D"]
    elif MolDirName == "H5O2pls":
        atomStr = ["O", "O", "H", "D", "D", "D", "D"]
    else:
        raise Exception("No atom list defined.")
    mol = Molecule(MoleculeName=MolDirName,
                   dimension=dimension,
                   atom_str=atomStr,
                   method=method,
                   scanCoords=scancoords,
                   embed_dict=embedDict,
                   OH=OH)  # this pulls log data from OO/OH scans
    return mol

def get_reducedmass():
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    mD = Constants.mass("D", to_AU=True)
    massdict = dict()
    muXH = ((2 * mO) * mH) / ((2 * mO) + mH)
    massdict["muXH"] = muXH
    muXD = ((2 * mO) * mD) / ((2 * mO) + mD)
    massdict["muXD"] = muXD
    muOH = 1/(1/mO + 1/mH)
    massdict["muOH"] = muOH
    muOO = mO / 2
    massdict["muOO"] = muOO
    return massdict

def g_mat_oo(x):
    """Calculates the constant G-matrix element for OO/OO"""
    g = (1/Constants.mass("O", to_AU=True)) + (1/Constants.mass("O", to_AU=True))
    test = np.repeat(g, x.shape[0])
    return test

def g_mat_oh(x):
    """Calculates the constant G-matrix element for OH/OH"""
    g = (1/(Constants.mass("O", to_AU=True))) + (1/Constants.mass("H", to_AU=True))
    test = np.repeat(g, x.shape[0])
    return test

def g_mat_offd(x):
    """Calculates the constant G-matrix element for OO/OH"""
    g = 1/Constants.mass("O", to_AU=True)
    test = np.repeat(g, x.shape[0])
    return test

def g_deriv(x):
    return np.zeros(x.shape[0])

def run_2D_DVR(moleculeObj, potential="scanPot", XHobj=None):
    """Runs 2D DVR over the original 2D potential"""
    from McUtils.Plots import ContourPlot
    from PotExpansions import ModelHarmonic
    from scipy import interpolate
    dvr_2D = DVR("ColbertMillerND")
    if potential == "scanPot":
        npz_filename = os.path.join(moleculeObj.mol_dir, "DVR Results", f"{moleculeObj.method}_2D_DVR_KC_OHPot.npz")
        twoD_grid = moleculeObj.logData.OHenergies
        xy = Constants.convert(twoD_grid[:, :2], "angstroms", to_AU=True)
        en = twoD_grid[:, 2]
        KC = True  # set KC = True anytime you use OH/OO
    elif potential == "xhPot":
        npz_filename = os.path.join(moleculeObj.mol_dir, "DVR Results", f"{moleculeObj.method}_2D_DVR_KC_XHPot.npz")
        twoD_grid = XHobj.logData.energies
        xy = Constants.convert(twoD_grid[:, :2], "angstroms", to_AU=True)
        en = twoD_grid[:, 2]
        KC = False
    elif potential == "harmPot":
        npz_filename = os.path.join(moleculeObj.mol_dir, "DVR Results", f"HMP_2D_DVR.npz")
        PotObj = ModelHarmonic(moleculeObj, CC=True)  # add "CC=True" to add in cubic coupling to potential
        en = PotObj.HarmonicPotential.flatten()
        xy = np.column_stack((PotObj.coord_grid[0].flatten(), PotObj.coord_grid[1].flatten()))
        KC = True  # set KC = True anytime you use OH/OO
    elif potential == "OHinXH":  # solves OH/OO grid using the XH scan potential
        npz_filename = os.path.join(moleculeObj.mol_dir, "DVR Results", f"OHgridinXHpot_2D_DVR.npz")
        xh_pot = XHobj.logData.energies  # electronic potential in XH/OO
        xhGrid = Constants.convert(xh_pot[:, :2], "angstroms", to_AU=True)  # XH/OO points in bohr
        twoD_grid = moleculeObj.logData.OHenergies
        xyoh = Constants.convert(twoD_grid[:, :2], "angstroms", to_AU=True)  # OO/OH grid points in bohr
        ohINxh_full = (xyoh[:, 0]/2) - xyoh[:, 1]  # convert OH coord to XH
        xhINoh_idx = np.argwhere((ohINxh_full >= np.min(xhGrid[:, 1])) * (ohINxh_full <= np.max(xhGrid[:, 1])))
        int_xy = np.column_stack((xyoh[xhINoh_idx, 0], ohINxh_full[xhINoh_idx]))
        # OO/OH grid in OO/XH coords WITH XH scan boundaries
        en = interpolate.griddata(xhGrid, xh_pot[:, 2], int_xy, method="cubic", fill_value=0.109)  # use interped OO/XH grid
        xy = int_xy  # use grid points you interpolate to, ie OH/OO becomes XH/OO
        KC = False
    elif potential == "XHinOH":  # solves XH/OO grid using the OH scan potential
        npz_filename = os.path.join(moleculeObj.mol_dir, "DVR Results", f"XHgridinOHpot_2D_DVR.npz")
        twoD_grid = moleculeObj.logData.OHenergies  # electronic potential in OH/OO
        ohGrid = Constants.convert(twoD_grid[:, :2], "angstroms", to_AU=True)  # OO/OH grid points in bohr
        xh_pot = XHobj.logData.energies
        xyxh = Constants.convert(xh_pot[:, :2], "angstroms", to_AU=True)  # OO/XH grid points in bohr
        xhINoh_full = (xyxh[:, 0]/2) - xyxh[:, 1]  # convert XH coord to OH
        xhINoh_idx = np.argwhere((xhINoh_full >= np.min(ohGrid[:, 1])) * (xhINoh_full <= np.max(ohGrid[:, 1])))
        int_xy = np.column_stack((xyxh[xhINoh_idx, 0], xhINoh_full[xhINoh_idx]))
        # OO/XH grid in OO/OH coords WITH OH scan boundaries
        en = interpolate.griddata(ohGrid, twoD_grid[:, 2], int_xy, method="cubic", fill_value=0.228)
        xy = int_xy  # use grid points you interpolate to, ie XH/OO becomes OH/OO
        KC = True  # set KC = True anytime you use OH/OO
    else:
        raise Exception(f"Can not use {potential} for this calculation.")
    en[en > 0.228] = 0.228  # sets threshold to 24000 cm^-1
    massdict = get_reducedmass()
    if KC:
        res = dvr_2D.run(potential_grid=np.column_stack((xy, en)),
                         divs=(100, 100), mass=[massdict["muOO"], massdict["muOH"]], g=[[g_mat_oo, g_mat_offd],
                                                                                        [g_mat_offd, g_mat_oh]],
                         g_deriv=[g_deriv, g_deriv], num_wfns=15,
                         domain=((min(xy[:, 0]),  max(xy[:, 0])), (min(xy[:, 1]), max(xy[:, 1]))),
                         results_class=ResultsInterpreter)
    else:
        res = dvr_2D.run(potential_grid=np.column_stack((xy, en)),
                         divs=(100, 100), mass=[massdict["muOO"], massdict["muXH"]], num_wfns=10,
                         domain=((min(xy[:, 0]),  max(xy[:, 0])), (min(xy[:, 1]), max(xy[:, 1]))),
                         results_class=ResultsInterpreter)
    # res.plot_potential(plot_class=ContourPlot, plot_units="wavenumbers", colorbar=True).show()
    dvr_grid = Constants.convert(res.grid, "angstroms", to_AU=False)
    dvr_pot = Constants.convert(res.potential_energy.diagonal(), "wavenumbers", to_AU=False)
    all_ens = Constants.convert(res.wavefunctions.energies, "wavenumbers", to_AU=False)
    print(all_ens)
    # ResultsInterpreter.wfn_contours(res)
    # oh1oo0 = int(input("OH=1 OO=0 Wavefunction Index: "))
    # oh1oo1 = int(input("OH=1 OO=1 Wavefunction Index: "))
    # oh1oo2 = int(input("OH=1 OO=2 Wavefunction Index: "))
    # ens = np.zeros(4)
    # wfns = np.zeros((4, res.wavefunctions[0].data.shape[0]))
    # for i, wf in enumerate(res.wavefunctions):
    #     wfn = wf.data
    #     if i == 0:
    #         wfns[0] = wfn
    #         ens[0] = all_ens[i]
    #     elif i == oh1oo0:
    #         wfns[1] = wfn
    #         ens[1] = all_ens[i]
    #     elif i == oh1oo1:
    #         wfns[2] = wfn
    #         ens[2] = all_ens[i]
    #     elif i == oh1oo2:
    #         wfns[3] = wfn
    #         ens[3] = all_ens[i]
    #     else:
    #         pass
    # # data saved in wavenumbers/angstroms
    # np.savez(npz_filename, grid=[dvr_grid], potential=[dvr_pot], vrwfn_idx=[0, oh1oo0, oh1oo1, oh1oo2],
    #          energy_array=ens, wfns_array=wfns)
    return npz_filename

if __name__ == '__main__':
    tetEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "outerO1": 5, "outerO2": 8, "inversion_atom": 8}
    triEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "xyPlane_atom": 5, "inversion_atom": 9}

    trimer = makeMolecule("H7O3pls", triEmbedDict, dimension="2D", OH=True)
    trimer_XH = makeMolecule("H7O3pls", triEmbedDict, dimension="2D", OH=False)
    tetramer = makeMolecule("H9O4pls", tetEmbedDict, dimension="2D", OH=True)
    tetramer_XH = makeMolecule("H9O4pls", tetEmbedDict, dimension="2D", OH=False)

    run_2D_DVR(tetramer, potential="harmPot", XHobj=tetramer_XH)

