import numpy as np
"""Still a big work in progress"""


class Potentials2D:
    def __init__(self):
        #  this should probably initialize an attribute so that you can call the returned function?
        pass

    def ho_2D(self, grid, k1=1, k2=1):
        """simple harmonic oscillator"""
        return k1 / 2 * np.power(grid[:, 0], 2) + k2 / 2 * np.power(grid[:, 1], 2)

    def uncoupled_int(self, xdat, ydat):
        """Takes 2 sets of 1D data points and creates an uncoupled 2D mesh
        :param xdat: (x, z) points along x coordinate
        :type xdat: ndarray
        :param ydat: (y, z) points along y coordinate
        :type ydat: ndarray
        :return: pf: function fit to grid points for evaluation.
        :rtype: function
        """
        from scipy import interpolate
        tck1 = interpolate.splrep(xdat[:, 0], xdat[:, 1], s=0)
        tck2 = interpolate.splrep(ydat[:, 0], ydat[:, 1], s=0)

        def pf(grid=None, x=None, y=None, tck1=tck1, tck2=tck2):
            if grid is not None:
                x = grid[:, 0]
                y = grid[:, 1]
            fit1 = interpolate.splev(x, tck1, der=0)
            fit2 = interpolate.splev(y, tck2, der=0)
            pvs = fit1 + fit2
            return pvs.flatten()
        return pf

    def exterp2d(self, data_points, vals):
        """creates a normal grid if input data is not then it extrapolates to input grid using the
            last value on either end.

        :param data_points:
        :type data_points:
        :return:
        :rtype:
        """
        from scipy import interpolate
        xx = np.unique(data_points[:, 0])
        yy = np.unique(data_points[:, 1])
        values = vals.reshape(len(yy), len(xx))  # MUST BE SHAPE NxM
        extrap_func = interpolate.interp2d(xx, yy, values, kind='cubic', fill_value=None)

        def pf(grid, extrap=extrap_func):
            x = np.unique(grid[:, 0])
            y = np.unique(grid[:, 1])
            pvs = extrap(x, y)
            return pvs.flatten()
        return pf


class Potentials1D:
    def __init__(self):
        #  this should probably initialize an attribute so that you can call the returned function?
        pass

    def ho(self, grid, k=1):
        return k / 2 * np.power(grid, 2)

    def harmonic(self, x=None, y=None):
        from scipy import interpolate
        tck = interpolate.splrep(x, y, k=2, s=0)

        def pf(grid, extrap=tck):
            y_fit = interpolate.splev(grid, extrap, der=0)
            return y_fit
        return pf

    def potlint(self, x=None, y=None):
        from scipy import interpolate
        tck = interpolate.splrep(x, y, s=0)

        def pf(grid, extrap=tck):
            y_fit = interpolate.splev(grid, extrap, der=0)
            return y_fit
        return pf
