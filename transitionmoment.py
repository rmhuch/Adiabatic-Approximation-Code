import numpy as np


def interp_dipoles(cut_array, OHDVRres):
    """Using the results of to_eck_struct_new, interpolates the dipole surface to be the len of the dvr wavefunctions.
    :param cut_array: big ol' array (col0:scancoord_1(ang), col1:scancoord_2(ang), col2:x-component,
            col3:y-component, col4:z-component, 3rd dimension: all different roos)
    :type cut_array: np.ndarray
    :return: interpolated dipole values (same shape as input)
    :rtype: np.ndarray
    """
    from scipy import interpolate
    if isinstance(cut_array, str):
        with open(cut_array) as data:
            val = eval(data.read())
    else:
        val = cut_array
    rohs = val[0, :, 1]
    dip_vecs = val[:, :, 2:]
    g = OHDVRres["potential"][0]
    num_pts = len(OHDVRres["wfns_array"])
    new_dip_vals = np.zeros((len(dip_vecs), num_pts, 5))
    for j in np.arange(3):
        for k, dip_vec in enumerate(dip_vecs):
            dip_vals = dip_vec[:, j]
            tck = interpolate.splrep(rohs, dip_vals, s=0)
            new_dip_vals[k, :, 0] = np.repeat(val[k, 0, 0], len(g))
            new_dip_vals[k, :, 1] = g
            new_dip_vals[k, :, j+2] = interpolate.splev(g, tck, der=0)
    return new_dip_vals


def psi_trans(interp_dip_array, OHDVRres):
    """ calculate full transition moment as a function of OO distance.
    :param interp_dip_array: big ol' array (col0:roo(ang), col1:roh(ang), col2:x-component,
            col3:y-component, col4:z-component, 3rd dimension: all different roos)
    :type interp_dip_array: np.ndarray
        :param OHDVRres:
    :type OHDVRres:
    :return: transition moments in each coordinate, row oriented. (row0: x, row1: y, row2: z)
    :rtype: np.ndarray
    """
    if isinstance(interp_dip_array, str):
        with open(interp_dip_array) as data:
            val = eval(data.read())
    else:
        val = interp_dip_array
    dip_vecs = val[:, :, 2:]
    mus = np.zeros((3, len(dip_vecs)))
    wavefunctions_array = OHDVRres["wfns_array"]
    for j in np.arange(3):
        for k in np.arange(len(dip_vecs)):
            # calculate transition moment
            gs_wfn = wavefunctions_array[k, :, 0]
            es_wfn = wavefunctions_array[k, :, 1]
            es_wfn_t = es_wfn.reshape(-1, 1)
            soup = np.diag(dip_vecs[k, :, j]).dot(es_wfn_t)
            mu = gs_wfn.dot(soup)
            mus[j, k] = mu
    return mus

