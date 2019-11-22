import numpy as np


def rot1(coords, dips, xAxis_atom=None):
    if xAxis_atom is None:
        raise Exception("No x-axis atom defined")
    # rotation of O(0) to x-axis
    # step 1: rotate about z-axis.
    y = coords[:, xAxis_atom, 1]
    x = coords[:, xAxis_atom, 0]
    phi_1 = np.arctan2(y, x)
    cphi = np.cos(phi_1)
    sphi = np.sin(phi_1)

    z_rotator = np.zeros((len(coords), 3, 3))
    z_rotator[:, 0, :] = np.column_stack((cphi, sphi, np.zeros(len(coords))))
    z_rotator[:, 1, :] = np.column_stack((-1*sphi, cphi, np.zeros(len(coords))))
    z_rotator[:, 2, :] = np.reshape(np.tile([0, 0, 1], len(coords)), (len(coords), 3))
    z_coord = np.matmul(z_rotator, coords.transpose(0, 2, 1)).transpose(0, 2, 1)
    z_dip = np.matmul(z_rotator, dips.transpose(0, 2, 1)).transpose(0, 2, 1)
    # step 2: rotate about y-axis.
    z = z_coord[:, xAxis_atom, 2]
    rho = z_coord[:, xAxis_atom, 0]
    phi_1p = np.arctan2(z, rho)
    cphi_1p = np.cos(phi_1p)
    sphi_1p = np.sin(phi_1p)
    y_rotator = np.zeros((len(z_coord), 3, 3))
    y_rotator[:, 0, :] = np.column_stack((cphi_1p, np.zeros(len(z_coord)), sphi_1p))
    y_rotator[:, 1, :] = np.reshape(np.tile([0, 1, 0], len(z_coord)), (len(z_coord), 3))
    y_rotator[:, 2, :] = np.column_stack((-1*sphi_1p, np.zeros(len(z_coord)), cphi_1p))
    y_coord = np.matmul(y_rotator, z_coord.transpose(0, 2, 1)).transpose(0, 2, 1)
    y_dip = np.matmul(y_rotator, z_dip.transpose(0, 2, 1)).transpose(0, 2, 1)
    return y_coord, y_dip


def rot2(coords, dips, xyPlane_atom=None, outerO1=None, outerO2=None):
    if xyPlane_atom is not None:
        z5 = coords[:, xyPlane_atom, 2]
        y5 = coords[:, xyPlane_atom, 1]
    elif outerO1 is not None and outerO2 is not None:
        # define bisector of other Os
        o1 = coords[:, outerO1, :]
        nrm1 = np.linalg.norm(o1, axis=1)
        onew = np.zeros((len(coords), 3))
        for i, row in enumerate(o1):
            onew[i, 0] = row[0]/nrm1[i]
            onew[i, 1] = o1[i, 1]/nrm1[i]
            onew[i, 2] = o1[i, 2]/nrm1[i]
        # ono = o1/np.linalg.norm(o1)
        o2 = coords[:, outerO2, :]
        nrm2 = np.linalg.norm(o2, axis=1)
        otwo = np.zeros((len(coords), 3))
        for i, row in enumerate(o1):
            otwo[i, 0] = row[0]/nrm2[i]
            otwo[i, 1] = o2[i, 1]/nrm2[i]
            otwo[i, 2] = o2[i, 2]/nrm2[i]
        bisector = (onew + otwo)
        z5 = bisector[:, 2]
        y5 = bisector[:, 1]
    else:
        raise Exception("rotation to xy-plane not defined")
    # rotation of either an O or the bisector of two O's to xy-plane by rotation about x-axis
    phi_3 = np.arctan2(z5, y5)
    cphi_3 = np.cos(phi_3)
    sphi_3 = np.sin(phi_3)
    x_rotator = np.zeros((len(coords), 3, 3))
    x_rotator[:, 0, :] = np.reshape(np.tile([1, 0, 0], len(coords)), (len(coords), 3))
    x_rotator[:, 1, :] = np.column_stack((np.zeros(len(coords)), cphi_3, sphi_3))
    x_rotator[:, 2, :] = np.column_stack((np.zeros(len(coords)), -1*sphi_3, cphi_3))
    x_coord = np.matmul(x_rotator, coords.transpose(0, 2, 1)).transpose(0, 2, 1)
    x_dip = np.matmul(x_rotator, dips.transpose(0, 2, 1)).transpose(0, 2, 1)
    return x_coord, x_dip


def inverter(coords, dips, inversion_atom=None):
    if inversion_atom is None:
        raise Exception("No inversion atom defined")
    coords[:, :, -1] *= np.sign(coords[:, inversion_atom, -1])[:, np.newaxis]
    dips[:, :, -1] *= np.sign(coords[:, inversion_atom, -1])[:, np.newaxis]
    return coords, dips


def many_rotations(log_data, centralO_atom=None, xAxis_atom=None, xyPlane_atom=None,
                   outerO1=None, outerO2=None, inversion_atom=None):
    from Converter import Constants
    all_coords = np.array(list(log_data.cartesians.values()))
    all_coords = Constants.convert(all_coords, "angstroms", to_AU=True)
    all_dips = np.array(list(log_data.dipoles.values()))
    all_dips = all_dips.reshape(len(all_coords), 1, 3)

    if centralO_atom is None:
        raise Exception("No origin atom defined")
    # shift to origin
    o_coords = all_coords - all_coords[:, np.newaxis, centralO_atom]
    o_dips = all_dips - all_coords[:, np.newaxis, centralO_atom]
    # rotation of O x-axis
    r1_coords, r1_dips = rot1(o_coords, o_dips, xAxis_atom)
    # rotation of other Os bisector to xy plane
    r2_coords, r2_dips = rot2(r1_coords, r1_dips, xyPlane_atom, outerO1, outerO2)
    # inversion of one O
    rot_coords, rot_dips = inverter(r2_coords, r2_dips, inversion_atom)
    dipadedodas = rot_dips.reshape(len(all_coords), 3)
    return rot_coords, dipadedodas

