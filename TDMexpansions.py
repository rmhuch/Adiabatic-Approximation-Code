import numpy as np

class TM2Dexpansion:
    @classmethod
    def cubicTDM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        delta_Roo = params["delta_Roo"]
        cubic_mus = np.zeros((*delta_Roo.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            cubic_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh + derivs[v[i]]["firstOO"]*delta_Roo + \
                           derivs[v[i]]["secondOH"]*(delta_roh**2)*(1/2) + \
                           derivs[v[i]]["secondOO"]*(delta_Roo**2)*(1/2) + \
                           derivs[v[i]]["mixedOHOO"]*delta_Roo*delta_roh + \
                           derivs[v[i]]["mixedOHOHOO"]*(delta_roh**2)*(1/2)*delta_Roo + \
                           derivs[v[i]]["mixedOHOOOO"]*delta_roh*(delta_Roo**2)*(1/2) + \
                           derivs[v[i]]["thirdOH"]*(delta_roh**3)*(1/3) + \
                           derivs[v[i]]["thirdOO"]*(delta_Roo**3)*(1/3)
        return cubic_mus

    @classmethod
    def quadTDM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        delta_Roo = params["delta_Roo"]
        quad_mus = np.zeros((*delta_Roo.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            quad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh + derivs[v[i]]["firstOO"]*delta_Roo + \
                           derivs[v[i]]["secondOH"]*(delta_roh**2)*(1/2) + \
                           derivs[v[i]]["secondOO"]*(delta_Roo**2)*(1/2) + \
                           derivs[v[i]]["mixedOHOO"]*delta_Roo*delta_roh
        return quad_mus

    @classmethod
    def quadBILINtdm(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        delta_Roo = params["delta_Roo"]
        biquad_mus = np.zeros((*delta_Roo.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            biquad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh + derivs[v[i]]["firstOO"]*delta_Roo + \
                           derivs[v[i]]["mixedOHOO"]*delta_Roo*delta_roh
        return biquad_mus

    @classmethod
    def quadOHtdm(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        delta_Roo = params["delta_Roo"]
        ohquad_mus = np.zeros((*delta_Roo.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            ohquad_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh + derivs[v[i]]["firstOO"]*delta_Roo + \
                           derivs[v[i]]["secondOH"]*(delta_roh**2)*(1/2)
        return ohquad_mus

    @classmethod
    def linTDM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        delta_Roo = params["delta_Roo"]
        lin_mus = np.zeros((*delta_Roo.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            lin_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh + derivs[v[i]]["firstOO"]*delta_Roo
        return lin_mus

    @classmethod
    def constTDM(cls, params, derivs):
        eqDip = params["eqDipole"]
        delta_roh = params["delta_roh"]
        lin_mus = np.zeros((*delta_roh.shape, 3))
        v = ['x', 'y', 'z']
        for i in np.arange(3):  # hard coded because this is always the number of components (ie x, y, z)
            lin_mus[:, i] = eqDip[i] + derivs[v[i]]["firstOH"]*delta_roh
        return lin_mus


class TM1Dexpansion:
    @classmethod
    def polyTDM(cls, x, poly_vals):
        poly_mus = np.zeros((len(x), 3))
        for i in np.arange(3):  # loop through components
            p = np.poly1d(poly_vals[i, :])
            poly_mus[:, i] = p(x)
        return poly_mus

    @classmethod
    def cubicTDM(cls, x, polyvals):
        cub_mus = np.zeros((len(x), 3))
        for i, v in enumerate(polyvals):  # loop through components
            cub_mus[:, i] = v[-4] * (x ** 3) + v[-3] * (x ** 2) + v[-2] * x + v[-1]
        return cub_mus

    @classmethod
    def quadTDM(cls, x, polyvals):
        quad_mus = np.zeros((len(x), 3))
        for i, v in enumerate(polyvals):  # loop through components
            quad_mus[:, i] = v[-3] * (x ** 2) + v[-2] * x + v[-1]
        return quad_mus

    @classmethod
    def linTDM(cls, x, polyvals):
        lin_mus = np.zeros((len(x), 3))
        for i, v in enumerate(polyvals):  # loop through components
            lin_mus[:, i] = v[-2] * x + v[-1]
        return lin_mus

    @classmethod
    def constTDM(cls, x, polyvals):
        const_mus = np.zeros((len(x), 3))
        for i, v in enumerate(polyvals):  # loop through components
            const_mus[:, i] = np.repeat(v[-1], len(x))
        return const_mus
