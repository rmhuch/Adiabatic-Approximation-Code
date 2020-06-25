import numpy as np
import os
from MolecularSys import *
from GaussianHandler import *
from Converter import Constants
"""Quick Script to pull dipoles from small scan around the minimum and compute the dipole derivatives for
 use in TDM expansions within the class based code. That codes calls to the files created by this script. """

def pull_data():
    from RunTests import makeMolecule
    tetEmbedDict = {"centralO_atom": 1, "xAxis_atom": 0, "outerO1": 5, "outerO2": 8, "inversion_atom": 8}
    molObj = makeMolecule("H9O4pls", tetEmbedDict)
    FDlog = os.path.join(molObj.mol_dir, "Finite Scan Data", "logs", "2D_finiteSPEtet_008_DCurrent.log")
    gaussdat = LogInterpreter(FDlog, moleculeObj=molObj)
    return molObj, gaussdat

def embed():
    molObj, gaussdat = pull_data()
    rot_coords = MolecularOperations(molObj, gaussdat).embeddedCoords
    return rot_coords

def calc_derivs(fd_ohs, fd_oos, FDgrid, FDvalues):
    from McUtils.Zachary import finite_difference
    derivs = dict()
    firstOH = finite_difference(fd_ohs, FDvalues[2, :], 1,
                                end_point_precision=0, stencil=5, only_center=True)[0]
    firstOO = finite_difference(fd_oos, FDvalues[:, 2], 1,
                                end_point_precision=0, stencil=5, only_center=True)[0]
    secondOH = finite_difference(fd_ohs, FDvalues[2, :], 2,
                                 end_point_precision=0, stencil=5, only_center=True)[0]
    secondOO = finite_difference(fd_oos, FDvalues[:, 2], 2,
                                 end_point_precision=0, stencil=5, only_center=True)[0]
    thirdOH = finite_difference(fd_ohs, FDvalues[2, :], 3,
                                end_point_precision=0, stencil=5, only_center=True)[0]
    thirdOO = finite_difference(fd_oos, FDvalues[:, 2], 3,
                                end_point_precision=0, stencil=5, only_center=True)[0]
    mixedOHOO = finite_difference(FDgrid, FDvalues, (1, 1), stencil=(5, 5),
                                  accuracy=0, end_point_precision=0, only_center=True)[0, 0]
    mixedOHOOOO = finite_difference(FDgrid, FDvalues, (1, 2), stencil=(5, 5),
                                    accuracy=0, end_point_precision=0, only_center=True)[0, 0]
    mixedOHOHOO = finite_difference(FDgrid, FDvalues, (2, 1), stencil=(5, 5),
                                    accuracy=0, end_point_precision=0, only_center=True)[0, 0]
    # convert derivs to XH coordinate
    derivs["firstOH"] = firstOH * -1
    derivs["firstOO"] = firstOO + (1 / 2) * firstOH
    derivs["secondOH"] = secondOH
    derivs["secondOO"] = secondOO + (1 / 4) * secondOH + (1 / 2) * mixedOHOO
    derivs["thirdOH"] = thirdOH * -1
    derivs["thirdOO"] = thirdOO + (1 / 8) * thirdOH + (3 / 4) * mixedOHOHOO + (3 / 2) * mixedOHOOOO
    derivs["mixedOHOO"] = secondOH * -1 + mixedOHOO * -1
    derivs["mixedOHOOOO"] = mixedOHOHOO * -1 + (-1 / 4) * thirdOH + mixedOHOHOO * -1
    derivs["mixedOHOHOO"] = mixedOHOHOO + (1 / 2) * thirdOH
    return derivs

def calc_coefs():
    # coords = np.load("FDH9O4pls_rotcoords.npy")
    dips = np.load(os.path.join(mainD, "structures", "FDH9O4pls_rotdips.npy"))
    molObj, gaussdat = pull_data()
    scancoords = np.array(list(gaussdat.cartesians.keys()))
    sort_ind = np.lexsort((scancoords[:, 1], scancoords[:, 0]))
    sort_grid = scancoords[sort_ind, :]
    sort_dips = dips[sort_ind, :] 

    fd_ohs = Constants.convert(np.unique(sort_grid[:, 1]), "angstroms", to_AU=True)
    fd_oos = Constants.convert(np.unique(sort_grid[:, 0]), "angstroms", to_AU=True)
    FDgrid = np.array(np.meshgrid(fd_oos, fd_ohs)).T
    FDvaluesx = np.reshape(sort_dips[:, 0], (5, 5))
    FDvaluesy = np.reshape(sort_dips[:, 1], (5, 5))
    FDvaluesz = np.reshape(sort_dips[:, 2], (5, 5))
    eqDipole = np.array((FDvaluesx[2, 2], FDvaluesy[2, 2], FDvaluesz[2, 2]))
    xderivs = calc_derivs(fd_ohs, fd_oos, FDgrid, FDvaluesx)
    yderivs = calc_derivs(fd_ohs, fd_oos, FDgrid, FDvaluesy)
    zderivs = calc_derivs(fd_ohs, fd_oos, FDgrid, FDvaluesz)
    derivs = {'x': xderivs, 'y': yderivs, 'z': zderivs}
    fn = os.path.join(mainD, "Finite Scan Data", "DipCoefsH9O4pls_smallscan.npz")
    np.savez(fn, x=xderivs, y=yderivs, z=zderivs, eqDip=eqDipole)
    print("eqDipole:", eqDipole)
    return derivs


if __name__ == '__main__':
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mainD = os.path.join(udrive, "H9O4pls")
    # embed()
    calc_coefs()
