import numpy as np

class Intensities:
    @classmethod
    def FranckCondon(cls, gs_wfn, es_wfn):
        intensities = np.zeros(es_wfn.shape[1])
        for i in np.arange(es_wfn.shape[1]):
            intensities[i] = np.dot(gs_wfn.T, es_wfn[:, i]) ** 2
        return intensities

    @classmethod
    def TDM(cls, gs_wfn, es_wfn, tdm):
        intensities = np.zeros(es_wfn.shape[1])
        comp_intents = np.zeros(3)
        for i in np.arange(es_wfn.shape[1]):  # excited state wfn
            for j in np.arange(3):  # transition moment component
                super_es = tdm[:, j].T * es_wfn[:, i]
                comp_intents[j] = np.dot(gs_wfn.T, super_es.T)
            intensities[i] = np.linalg.norm(comp_intents) ** 2
        return intensities

    @classmethod
    def TwoD(cls, wfns, tdm, gridpoints=None):
        import matplotlib.pyplot as plt
        intensities = np.zeros(len(wfns)-1)
        for i in np.arange(1, len(wfns)):  # starts at 1 to only loop over exciting states
            x = 0
            for j in np.arange(3):
                x += (np.dot(wfns[0], (tdm[:, j] * wfns[i]))) ** 2
                # y = tdm[:, j]
                # if gridpoints is not None:
                #     grid = gridpoints[0].transpose(2, 0, 1)
                #     plt.contourf(*grid, y.reshape((100, 100)))
                #     plt.title(f" {i}, {j}")
                #     plt.colorbar()
                #     plt.show()
                # x is magnitude squared because instead of taking the sqrt and then squaring, I just didn't.
            intensities[i-1] = x
        return intensities

