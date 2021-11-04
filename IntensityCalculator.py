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
            print("excited state: ", i)
            for j in np.arange(3):  # transition moment component
                super_es = tdm[:, j].T * es_wfn[:, i]
                comp_intents[j] = np.dot(gs_wfn.T, super_es.T)
                print(j, comp_intents[j])
            intensities[i] = np.linalg.norm(comp_intents) ** 2
        return intensities

    @classmethod
    def TwoD(cls, wfns, tdm, gridpoints=None):
        import matplotlib.pyplot as plt
        intensities = np.zeros(len(wfns)-1)
        comp_intents = np.zeros(3)
        for i in np.arange(1, len(wfns)):  # starts at 1 to only loop over exciting states
            print("excited state: ", i)
            for j in np.arange(3):
                super_es = tdm[:, j] * wfns[i]
                comp_intents[j] = np.dot(wfns[0], super_es)
                print(j, comp_intents[j])
            intensities[i-1] = np.linalg.norm(comp_intents) ** 2
        return intensities

