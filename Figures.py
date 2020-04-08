import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants


class AnnePlots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.OHDVRresults = np.load(OHDVRnpz)
        self.logData = moleculeObj.logData

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
        self.logData = moleculeObj.logData
        import os
        self.fig_dir = os.path.join(moleculeObj.mol_dir, "figures")

    def make_scan_plots(self, grid=False, contour=True):
        plt.rcParams.update({'font.size': 16})
        if grid:
            pts = np.array(list(self.logData.cartesians.keys()))
            plt.plot(pts[:, 0], pts[:, 1], 'ok', markersize=1.5)
            # plt.close()
        if contour:
            pts = self.logData.rawenergies
            pot = pts[:, 2]
            potwv = Constants.convert(pot, "wavenumbers", to_AU=False)
            potwv[potwv > 24000] = 24000
            plt.tricontourf(pts[:, 0], pts[:, 1], potwv, cmap='Purples_r', levels=8)
            plt.colorbar()
            plt.tricontour(pts[:, 0], pts[:, 1], potwv, colors='k', levels=8)
            # plt.close()

    @staticmethod
    def wfn_plots(i, potz, wfns2plt, wfns, ens, colors):
        plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=6.0)
        for k in range(wfns2plt):
            plt.plot(potz[i, :, 0], (wfns[i, :, k] * 5000) + ens[i, k + 1], colors[k], linewidth=4.0)
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
            if self.OHDVRresults["method"] == "harm":
                plt.plot(potz[i, :, 0], potz[i, :, 1], '-m', linewidth=6.0)
            else:
                plt.plot(potz[i, :, 0], potz[i, :, 1], '-k', linewidth=6.0)
            # minIdx = np.argmin(potz[i, :, 1])
            # print(f"{j} : min {potz[i, minIdx, 0]}")
            for k in range(wfns2plt):
                plt.plot(potz[i, :, 0], (wfns[i, :, k] * 6000) + eps[i, (k + 1)], colors[k], linewidth=4.0)
            plt.ylim(0, 15000)
            # plt.xlim(0.75, 1.6)
            plt.title(f"Roo = {j}")
            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/{self.molecule.method}_{self.OHDVRresults['method']}_OHwfns_Roo_{j}.png")
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
                    plt.plot(potz[i, :, 0], (wfns[i, :, k] * 1000) + energies[i, (k + 1)], colors[k + 1], linewidth=4.0)
        plt.ylim(0, 8000)
        plt.xlim(2, 4)
        # plt.title(f"OH = {i}")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.molecule.method}_OOwfns_{self.OHDVRresults['method']}.png")
        plt.close()

    def make_adiabatplots(self):
        from scipy import interpolate, optimize
        import os
        mini_pot = self.logData.minimum_pot()
        eps = self.OHDVRresults["epsilonPots"]
        oo_energies = self.OODVRresults["energy_array"]
        fig = plt.figure(figsize=(6, 6), dpi=300)
        plt.rcParams.update({'font.size': 16})
        grid = self.OODVRresults["potential"][0][:, 0]
        # plot electronic energy
        E = Constants.convert(mini_pot[:, 2], "wavenumbers", to_AU=False).T
        roos = mini_pot[:, 0]
        tck = interpolate.splrep(roos, E, s=0)
        E_fit = interpolate.splev(grid, tck, der=0)
        plt.plot(grid, E_fit, '-k', linewidth=6.0)
        # plot epsilon curves and energy levels
        colors = ["royalblue", "crimson"]
        for i in range(2):  # plot curves
            pot = eps[:, i + 1]
            en_level = oo_energies[i, :]
            print(f"{self.OHDVRresults['method']} Frequency OO(OH={i}): ", en_level[1] - en_level[0])
            print(f"{self.OHDVRresults['method']} Ground State(OH={i}): ", en_level[0])
            tck = interpolate.splrep(eps[:, 0], pot, s=0)
            pot_fit = interpolate.splev(grid, tck, der=0)
            plt.plot(grid, pot_fit, colors[i], linewidth=6.0)
            for j in range(2):  # plot levels
                # -- if levels aren't plotting check xl and ar and make sure they are making actual vectors.
                xl = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[0], method="lm").x
                xr = optimize.root(lambda x: interpolate.splev(x, tck, der=0) - en_level[j], grid[-1], method="lm").x
                enl_x = np.linspace(xr[0], xl[0], 10)
                E = [en_level[j]] * len(enl_x)
                plt.plot(enl_x, E, colors[i], linewidth=4.0)
        print(f"{self.OHDVRresults['method']} Frequency OH: ", oo_energies[1, 0] - oo_energies[0, 0])
        plt.title(f"{self.molecule.method} {self.OHDVRresults['method']} OH")
        plt.ylim(0, 8000)
        plt.xlim(2, 4)
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.molecule.method}_adiabatplot_{self.OHDVRresults['method']}OH.png")
        plt.close()


class TMplots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, OODVRnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.OHDVRnpz = OHDVRnpz
        self.OODVRnpz = OODVRnpz
        self.OODVRres = np.load(OODVRnpz)
        import os
        self.fig_dir = os.path.join(moleculeObj.mol_dir, "figures")
        self._tmObj = None

    @property
    def tmObj(self):
        if self._tmObj is None:
            from transitionmoment import TransitionMoment
            self._tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D",
                                           OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
        return self._tmObj

    def DipoleSurfaces(self, preEmbed=False):
        """Plots the x,y,z components of the dipole surface."""
        from McUtils.Plots import GraphicsGrid, ContourPlot, ListContourPlot
        dip_struct = self.tmObj.makeDipStruct(preEmbed=preEmbed)
        # forAnne = dip_struct.reshape((342, 5))
        # forAnne[:, 0] = forAnne[:, 0] - forAnne[90, 0]
        # forAnne[:, 1] = forAnne[:, 1] - forAnne[4, 1]
        # forAnne[:, :2] = Constants.convert(forAnne[:, :2], "angstroms", to_AU=True)
        # np.savetxt("OHOO_Dipoles_T.txt", forAnne)
        plt.close()
        roos = dip_struct[:, 0, 0]
        rohs = dip_struct[0, :, 1]
        dip_vecs = dip_struct[:, :, 2:]
        mini = np.amin(dip_vecs)
        maxi = np.amax(dip_vecs)
        comp = ['X', 'Y', 'Z']
        main = GraphicsGrid(ncols=3, nrows=1)
        main.image_size = (1000, 400)
        for i, dip in enumerate(dip_vecs.T):
            opts = dict(
                plot_style=dict(cmap="Purples_r", levels=10, vmin=mini, vmax=maxi),
                figure=main[0, i],
                axes_labels=['OO bond Distance ($\mathrm{\AA}$)', 'OH bond Distance ($\mathrm{\AA}$)'])
            main[0, i] = ContourPlot(roos, rohs, dip, **opts)
            main[0, i].plot_label = f'{comp[i]}-Component of Dipole'
        main.colorbar = {"graphics": main[0, 0].graphics}
        plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_dipoleplots.png")
        plt.close()

    def InterpolatedDips(self):
        """will plot dipoles (dots) and interpolation results (lines) for checking"""
        dip_struct = self.tmObj.makeDipStruct()
        roos = dip_struct[:, 0, 0]
        interped_dips = self.tmObj.interp_dipoles()
        color = ["blue", "orange", "green"]
        comp = ["X", "Y", "Z"]
        plt.rcParams.update({'font.size': 16})
        for k, oo in enumerate(roos):
            for j in np.arange(3):
                plt.plot(interped_dips[k, :, 1], interped_dips[k, :, j + 2],
                         label=f"{comp[j]}-Component", color=color[j], linewidth=4.0)
                # plt.plot(dip_struct[k, :, 1], dip_struct[k, :, j+2], "o", label=f"{comp[j]}-Component", color=color[j])
            plt.title(f"Roo = {oo}")
            plt.xlabel('OH bond Distance ($\mathrm{\AA}$)')
            plt.ylabel('Dipole')
            plt.xlim(0.75, 1.6)
            plt.legend()
            plt.show()
            plt.close()

    def TransitionMoments(self, color=None, ylim=None):
        if color is None:
            color = ["blue", "orange", "green"]
        comp = ["X", "Y", "Z"]
        x = self.tmObj.mus[0][:, 0, 0]
        mus = self.tmObj.mus[1]
        interp_x = self.tmObj.tdms[0]
        poly_tdm = self.tmObj.tdms[1]["poly"]
        for i in range(3):
            plt.plot(x, mus[:, i], "o", label=f"{comp[i]}-Component", color=color[i])
            plt.plot(interp_x, poly_tdm[:, i], color=color[i])
        plt.ylabel("Transition Moment")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_polyTDM.png")
        plt.close()

    def componentTMs(self, ylim=None):
        comp = ["X", "Y", "Z"]
        x = self.tmObj.mus[0][:, 0, 0]
        mus = self.tmObj.mus[1]
        bigGrid = self.tmObj.tdms[0]
        exMus = self.tmObj.tdms[1]
        for i, v in enumerate(comp):
            plt.plot(x, mus[:, i], 'ok')  # plot calculated TMs
            plt.plot(bigGrid, exMus["poly"][:, i], '--b', label=f'6th-Order')  # plot n order polynomial
            plt.plot(bigGrid, exMus["cubic"][:, i], '-r', label='cubic')  # plot cubic expansion
            plt.plot(bigGrid, exMus["quad"][:, i], '-c', label='quad')  # plot quadratic expansion
            plt.plot(bigGrid, exMus["lin"][:, i], '-m', label='linear')  # plot linear expansion
            plt.plot(bigGrid, exMus["const"][:, i], '-y', label='constant')  # plot constant term
            if ylim is not None:
                plt.ylim(*ylim)
            plt.legend()
            plt.title(f"{comp[i]} Component of TDM")
            plt.xlabel("OO distance")
            plt.ylabel("Transition Moment")
            plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_{comp[i]}componentTDM.png")
            plt.close()

class TM2Dplots:
    def __init__(self, moleculeObj=None, TwoDnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.TwoDnpz = TwoDnpz
        import os
        self.fig_dir = os.path.join(moleculeObj.mol_dir, "figures")
        self._tmObj = None

    @property
    def tmObj(self):
        if self._tmObj is None:
            from transitionmoment import TransitionMoment
            self._tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="2D", TwoDnpz=self.TwoDnpz)
        return self._tmObj

    def DipoleSurfaces(self):
        from McUtils.Plots import GraphicsGrid, ListContourPlot
        grid = self.tmObj.TwoDDips[0]
        dips = self.tmObj.TwoDDips[1]
        # grid[:, 0] = grid[:, 0] - grid[2900, 0]
        # grid[:, 1] = grid[:, 1] - grid[46, 1]
        # grid = Constants.convert(grid, "angstroms", to_AU=True)
        # forAnne = np.column_stack((grid[:, 0], grid[:, 1], dips))
        # np.savetxt("XHOO_Dipoles.txt", forAnne)
        mini = np.amin(dips)
        maxi = np.amax(dips)
        comp = ['X', 'Y', 'Z']
        main = GraphicsGrid(ncols=3, nrows=1)
        main.image_size = (1000, 400)
        for i in np.arange(dips.shape[1]):
            opts = dict(
                plot_style=dict(cmap="Purples_r", levels=10, vmin=mini, vmax=maxi),
                figure=main[0, i],
                axes_labels=['OO bond Distance ($\mathrm{\AA}$)', 'OH bond Distance ($\mathrm{\AA}$)'])
            main[0, i] = ListContourPlot(np.column_stack((grid[:, 0], grid[:, 1], dips[:, i])), **opts)
            main[0, i].plot_label = f'{comp[i]}-Component of Dipole'
        main.colorbar = {"graphics": main[0, 0].graphics}
        plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_2D_dipoleplots.png")
        plt.close()

    def componentTMs(self):
        from McUtils.Plots import GraphicsGrid, ListContourPlot
        comp = ["X", "Y", "Z"]
        Grid = self.tmObj.TwoDtdms[0]
        exMus = self.tmObj.TwoDtdms[1]
        mini = np.amin(exMus["poly"])
        maxi = np.amax(exMus["poly"])
        for i in np.arange(3):  # make one figure per comp
            main = GraphicsGrid(ncols=2, nrows=2)
            main.image_size = (1000, 1000)
            opts = dict(
                plot_style=dict(cmap="Purples_r", levels=10, vmin=mini, vmax=maxi),
                axes_labels=['OO bond Distance ($\mathrm{\AA}$)', 'OH bond Distance ($\mathrm{\AA}$)'])
            main[0, 0] = ListContourPlot(np.column_stack((Grid[:, 0], Grid[:, 1], exMus["poly"][:, i])),
                                         figure=main[0, 0], **opts)
            main[0, 0].plot_label = f'{comp[i]}-Component 2D Dipoles'
            main[0, 1] = ListContourPlot(np.column_stack((Grid[:, 0], Grid[:, 1], exMus["cubic"][:, i])),
                                         figure=main[0, 1], **opts)
            main[0, 1].plot_label = f'{comp[i]}-Component Cubic TDM'
            main[1, 0] = ListContourPlot(np.column_stack((Grid[:, 0], Grid[:, 1], exMus["quad"][:, i])),
                                         figure=main[1, 0], **opts)
            main[1, 0].plot_label = f'{comp[i]}-Component Quadratic TDM'
            main[1, 1] = ListContourPlot(np.column_stack((Grid[:, 0], Grid[:, 1], exMus["lin"][:, i])),
                                         figure=main[1, 1], **opts)
            main[1, 1].plot_label = f'{comp[i]}-Component Linear TDM'
            main.colorbar = {"graphics": main[0, 0].graphics}
            plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_2D_{comp[i]}_TDMexpansions.png")
            plt.close()


