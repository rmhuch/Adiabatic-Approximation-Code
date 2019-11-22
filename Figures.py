import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants


class AAplots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, OODVRnpz=None, **kwargs):
        self.params = kwargs
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.OHDVRresults = np.load(OHDVRnpz)
        self.OODVRresults = np.load(OODVRnpz)
        self._logData = None

    @property
    def logData(self):
        if self._logData is None:
            from GaussianHandler import LogInterpreter
            if self.molecule.method == "rigid":
                optBool = False
            else:
                optBool = True
            self._logData = LogInterpreter(*self.molecule.scanLogs, moleculeObj=self.molecule, optimized=optBool)
        return self._logData

    @staticmethod
    def wfn_plots(i, potz, wfns2plt, wfns, ens, colors):
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=6.0)
        for k in range(wfns2plt):
            plt.plot(potz[i, :, 0], (wfns[i, :, k]*5000) + ens[i, k+1], colors[k], linewidth=4.0)
        plt.tight_layout()
        return fig

    def ohWfn_plots(self, wfns2plt=4, **params):
        potz = self.OHDVRresults["potential"]
        wfns = self.OHDVRresults["wfns_array"]
        eps = self.OHDVRresults["epsilonPots"]
        roos = eps[:, 0]
        colors = ["royalblue", "crimson", "violet", "orchid", "plum", "hotpink"]
        for i, j in enumerate(roos):
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(6, 6), dpi=300)
            plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=6.0)
            for k in range(wfns2plt):
                plt.plot(potz[i, :, 0], (wfns[i, :, k] * 5000) + eps[i, (k + 1)], colors[k], linewidth=4.0)
            plt.ylim(0, 25000)
            plt.title(f"Roo = {j}")
            plt.tight_layout()
            plt.savefig(f"{self.molecule.method}_{self.OHDVRresults['method']}_OHwfns_Roo_{j}.png")
            plt.close()

    def ooWfn_plots(self, wfns2plt=2, **params):
        potz = self.OODVRresults["potential"]  # angstroms/wavenumbers
        wfns = self.OODVRresults["wfns_array"]
        energies = self.OODVRresults["energy_array"]  # wavenumbers
        colors = ["royalblue", "darkmagenta", "mediumslateblue", "mediumblue"]
        oh_colors = ["royalblue", "crimson"]
        plt.figure(figsize=(6, 6), dpi=300)
        for i in range(len(potz)):
            plt.rcParams.update({'font.size': 20})
            # plt.figure(figsize=(6, 6), dpi=300)
            plt.plot(potz[i, :, 0], potz[i, :, 1], oh_colors[i], linewidth=6.0)
            if i == 0:
                plt.plot(potz[i, :, 0], (wfns[i, :, 0] * 1000) + energies[i, 1], colors[0], linewidth=4.0)
            else:
                for k in range(wfns2plt):
                    plt.plot(potz[i, :, 0], (wfns[i, :, k] * 1000) + energies[i, (k + 1)], colors[k+1], linewidth=4.0)
        plt.ylim(0, 8000)
        plt.xlim(2, 4)
        # plt.title(f"OH = {i}")
        plt.tight_layout()
        plt.savefig(f"{self.molecule.method}_OOwfns_{self.OHDVRresults['method']}.png")
        plt.close()

    def make_adiabatplots(self):
        from scipy import interpolate, optimize
        mini_pot = self.logData.minimum_pot()
        eps = self.OHDVRresults["epsilonPots"]
        oo_energies = self.OODVRresults["energy_array"]
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.rcParams.update({'font.size': 20})
        grid = self.OODVRresults["potential"][0]
        # plot electronic energy
        E = Constants.convert(mini_pot[:, 2], "wavenumbers", to_AU=False).T
        roos = mini_pot[:, 0]
        tck = interpolate.splrep(roos, E, s=0)
        E_fit = interpolate.splev(grid, tck, der=0)
        plt.plot(grid, E_fit, '-k', linewidth=6.0)
        # plot epsilon curves and energy levels
        colors = ["royalblue", "crimson"]
        for i in range(2):  # plot curves
            pot = eps[:, i+1]
            en_level = oo_energies[i, :]
            print(f"Energy diff (OH={i}): ", en_level[1]-en_level[0])
            tck = interpolate.splrep(eps[:, 0], pot, s=0)
            pot_fit = interpolate.splev(grid, tck, der=0)
            plt.plot(grid, pot_fit, colors[i], linewidth=6.0)
            for j in range(2):  # plot levels
                # -- if levels aren't plotting check xl and ar and make sure they are making actual vectors.
                # xl = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[0], method="lm").x
                xr = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[-1], method="lm").x
                enl_x = np.linspace(xr[0], xr[-1], 10)
                E = [en_level[j]] * len(enl_x)
                plt.plot(enl_x, E, colors[i], linewidth=4.0)

        plt.title(f"{self.molecule.method} {self.OHDVRresults['method']} OH")
        plt.ylim(0, 8000)
        plt.xlim(2, 4)
        plt.tight_layout()
        plt.savefig(f"{self.molecule.method}_adiabatplot_{self.OHDVRresults['method']}OH.png")
        plt.close()


class TMplots:
    def __init__(self):
        pass

    def InterpolatedDips(self):
        """will plot dipoles (dots) and interpolation results (lines) for checking"""
        plt.plot(g, new_dip_vals[k, :, j + 2])
        plt.plot(rohs, dip_vals, 'o')
        plt.xlabel('OH bond Distance ($\mathrm{\AA}$)')
        plt.ylabel('Dipole UNITS')
        plt.show()
        plt.close()

    def TransitionMoments(self):
        plt.plot(roos, mus[0, :], label="X-Component")
        plt.plot(roos, mus[1, :], label="Y-Component")
        plt.plot(roos, mus[2, :], label="Z-Component")
        plt.xlabel("OO distance")
        plt.ylabel("Transition Moment")
        plt.ylim(-0.75, 0.25)
        plt.legend()
        plt.show()