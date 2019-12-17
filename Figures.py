import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants


class AnnePlots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.OHDVRresults = np.load(OHDVRnpz)
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

    def eqOHPlot(self, color="k"):
        plt.rcParams.update({'font.size': 18})
        mini_pot = self.logData.minimum_pot()
        plt.plot(mini_pot[:, 0], mini_pot[:, 1], 'o', c=color,
                 label=f"{self.molecule.MoleculeName} {self.molecule.method}")
        plt.plot(mini_pot[:, 0], mini_pot[:, 1], c=color, linewidth=2.5)
        plt.xlabel('OO bond Distance ($\mathrm{\AA}$)')
        plt.ylabel('OH equilibrium bond Distance ($\mathrm{\AA}$)')
        plt.tight_layout()

    def freqOHPlot(self, color="k"):
        plt.rcParams.update({'font.size': 18})
        eps = self.OHDVRresults["epsilonPots"]
        roos = eps[:, 0]
        energies = eps[:, 2] - eps[:, 1]
        plt.plot(roos, energies, 'o', c=color,
                 label=f"{self.molecule.MoleculeName} {self.OHDVRresults['method']} {self.molecule.method}")
        plt.plot(roos, energies, '-', c=color, linewidth=2.5)
        plt.xlabel('OO bond Distance ($\mathrm{\AA}$)')
        plt.ylabel('OH Frequency ($\mathrm{cm^-1}$)')
        plt.tight_layout()


class AAplots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, OODVRnpz=None, **kwargs):
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

    def ohWfn_plots(self, wfns2plt=4, **kwargs):
        potz = self.OHDVRresults["potential"]
        wfns = self.OHDVRresults["wfns_array"]
        eps = self.OHDVRresults["epsilonPots"]
        roos = eps[:, 0]
        colors = ["royalblue", "crimson", "violet", "orchid", "plum", "hotpink"]
        for i, j in enumerate(roos):
            plt.rcParams.update({'font.size': 20})
            plt.figure(figsize=(6, 6), dpi=300)
            plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=6.0)
            # minIdx = np.argmin(potz[i, :, 1])
            # print(f"{j} : min {potz[i, minIdx, 0]}")
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
        import os
        mini_pot = self.logData.minimum_pot()
        eps = self.OHDVRresults["epsilonPots"]
        oo_energies = self.OODVRresults["energy_array"]
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.rcParams.update({'font.size': 20})
        grid = self.OODVRresults["potential"][0][:, 0]
        # plot electronic energy
        E = Constants.convert(mini_pot[:, 2], "wavenumbers", to_AU=False).T
        roos = mini_pot[:, 0]
        tck = interpolate.splrep(roos, E, s=0)
        E_fit = interpolate.splev(grid, tck, der=0)
        plt.plot(grid, E_fit, '-k', linewidth=6.0)
        # plot epsilon curves and energy levels
        colors = ["royalblue", "crimson"]
        minOH = np.zeros(2)
        for i in range(2):  # plot curves
            pot = eps[:, i+1]
            en_level = oo_energies[i, :]
            print(f"{self.OHDVRresults['method']} Frequency OO(OH={i}): ", en_level[1]-en_level[0])
            print(f"{self.OHDVRresults['method']} Ground State(OH={i}): ", en_level[0])
            tck = interpolate.splrep(eps[:, 0], pot, s=0)
            pot_fit = interpolate.splev(grid, tck, der=0)
            minOH[i] = np.min(pot_fit)
            plt.plot(grid, pot_fit, colors[i], linewidth=6.0)
            for j in range(2):  # plot levels
                # -- if levels aren't plotting check xl and ar and make sure they are making actual vectors.
                xl = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[0], method="lm").x
                xr = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[-1], method="lm").x
                enl_x = np.linspace(xr[0], xl[0], 10)
                E = [en_level[j]] * len(enl_x)
                plt.plot(enl_x, E, colors[i], linewidth=4.0)
        print(f"{self.OHDVRresults['method']} Frequency OH: ", minOH[1] - minOH[0])
        plt.title(f"{self.molecule.method} {self.OHDVRresults['method']} OH")
        plt.ylim(0, 8000)
        plt.xlim(2, 4)
        plt.tight_layout()
        figDir = os.path.join(self.molecule.mol_dir, "figures")
        plt.savefig(os.path.join(figDir, f"{self.molecule.method}_adiabatplot_{self.OHDVRresults['method']}OH.png"))
        plt.close()


class TMplots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, OODVRnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")

        self.OHDVRnpz = OHDVRnpz
        self.OODVRnpz = OODVRnpz
        self.OODVRres = np.load(OODVRnpz)
        self._logData = None
        self.scanGrid = self.molecule.scanGrid
        from transitionmoment import TransitionMoment
        self.tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D", OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
    #
    # @property
    # def tmObj(self):
    #     if self._tmObj is None:
    #         from transitionmoment import TransitionMoment
    #         self._tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D",
    #                                        OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
    #     return self._tmObj

    def DipoleSurfaces(self, preEmbed=False):
        """Plots the x,y,z components of the dipole surface."""
        from McUtils.Plots import GraphicsGrid, ContourPlot
        dip_struct = self.tmObj.makeDipStruct(preEmbed=preEmbed)
        roos = dip_struct[:, 0, 0]
        rohs = dip_struct[0, :, 1]
        dip_vecs = dip_struct[:, :, 2:]
        min = np.amin(dip_vecs)
        max = np.amax(dip_vecs)
        comp = ['X', 'Y', 'Z']
        main = GraphicsGrid(ncols=3, nrows=1)
        main.image_size = (1000, 400)
        for i, dip in enumerate(dip_vecs.T):
            opts = dict(
                plot_style=dict(cmap="plasma", levels=10, vmin=min, vmax=max),
                figure=main[0, i],
                axes_labels=['OO bond Distance ($\mathrm{\AA}$)', 'OH bond Distance ($\mathrm{\AA}$)'])
            main[0, i] = ContourPlot(roos, rohs, dip, **opts)
            main[0, i].plot_label = f'{comp[i]}-Component of Dipole'
        main.colorbar = {"graphics": main[0, 0].graphics}
        # plt.tight_layout()
        plt.savefig(f"{self.molecule.method}_dipoleplots.png")
        plt.close()

    def InterpolatedDips(self):
        """will plot dipoles (dots) and interpolation results (lines) for checking"""
        dip_struct = self.tmObj.makeDipStruct()
        roos = dip_struct[:, 0, 0]
        interped_dips = self.tmObj.interp_dipoles()
        for k, oo in enumerate(roos):
            for j in np.arange(3):
                plt.plot(interped_dips[k, :, 1], interped_dips[k, :, j+2])
                plt.plot(dip_struct[k, :, 1], dip_struct[k, :, j+2], 'o')
            plt.title(f"Roo = {oo}")
            plt.xlabel('OH bond Distance ($\mathrm{\AA}$)')
            plt.ylabel('Dipole UNITS')
            plt.show()
            plt.close()

    def TransitionMoments(self, color=None, ylim=None):
        if color is None:
            color = ["blue", "orange", "green"]
        comp = ["X", "Y", "Z"]
        for i in range(3):
            plt.plot(self.tmObj.mus[0][:, 0], self.tmObj.mus[0][:, i+1], "o", label=f"{comp[i]}-Component", color=color[i])
            plt.plot(self.tmObj.mus[1][:, 0], self.tmObj.mus[1][:, i+1], color=color[i])
        ooWfns = self.OODVRres["wfns_array"]
        oopot = self.OODVRres["potential"]
        wfnAmpIdx = np.argwhere(ooWfns[0, :, 0] > 1E-5)
        ooWfnAmp = oopot[0, wfnAmpIdx, 0]
        print(np.min(ooWfnAmp), np.max(ooWfnAmp))
        plt.axvspan(np.min(ooWfnAmp), np.max(ooWfnAmp), facecolor="#2ca02c", alpha=0.3)
        # plt.plot(oopot[0, :, 0], ooWfns[0, :, 0], '-k')
        # plt.plot(oopot[1, :, 0], ooWfns[1, :, 0]+0.5, '-k')
        plt.xlabel("OO distance")
        plt.ylabel("Transition Moment")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        # plt.show()
