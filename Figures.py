import numpy as np
import matplotlib.pyplot as plt
from Converter import Constants

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


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
        if "AnharmOH" in OHDVRnpz:
            self.OHDVR = "anharm"
        else:
            self.OHDVR = "harm"
        self.OODVRresults = np.load(OODVRnpz)
        if "harmOODVR" in OODVRnpz:
            self.OODVR = "harm"
        else:
            self.OODVR = "anharm"
        self.logData = moleculeObj.logData
        import os
        self.fig_dir = os.path.join(moleculeObj.mol_dir, "figures")
        self.wfn_dir = os.path.join(moleculeObj.mol_dir, "DVR Results", "wfns")

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
            plt.tricontourf(pts[:, 0], pts[:, 1], potwv, cmap='viridis', levels=8)
            plt.colorbar()
            plt.tricontour(pts[:, 0], pts[:, 1], potwv, colors='k', levels=8)
            plt.show()

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
            plt.savefig(f"{self.wfn_dir}/{self.molecule.method}_{self.OHDVR}_OHwfns_Roo_{j}.png")
            plt.close()

    def ohWfn_PAs(self, **kwargs):
        from matplotlib.lines import Line2D
        potz = self.OHDVRresults["potential"]
        wfns = self.OHDVRresults["wfns_array"]
        eps = self.OHDVRresults["epsilonPots"]
        roos = eps[:, 0]
        colors = ["grey", "red", "orange", "deeppink"]
        for i, j in enumerate(roos):
            fig = plt.figure(facecolor="white")
            ax1 = plt.axes()
            ax1.get_xaxis().tick_bottom()
            ax1.axes.get_yaxis().set_visible(False)
            for k in np.arange(2):
                plt.plot(potz[i, :, 0], (wfns[i, :, k]**2), linewidth=3.0, color=colors[k+1], label="$\psi_{%d_{XH}}$" % k)
            plt.ylim(-0.001, 0.025)
            plt.xlim(-0.4, 0.7)
            # ax1.add_artist(Line2D((-0.4, 0.7), (-0.001, -0.001), color='k', linewidth=2))
            plt.xlabel("$\mathrm{r_{XH}}$ ($\mathrm{\AA}$)", size=16)
            # plt.ylabel("Probability Amplitude", size=16)
            plt.legend(fontsize=14)
            plt.title(f"Roo = {j}", size=20)
            plt.tight_layout()
            plt.savefig(f"{self.wfn_dir}/{self.molecule.method}_{self.OHDVR}_OHPAs_Roo_{j}_test.png")
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
        plt.savefig(f"{self.fig_dir}/{self.molecule.method}_OOwfns_{self.OHDVR}OH{self.OODVR}OO.png")
        plt.close()

    def make_adiabatplots(self):
        from scipy import interpolate, optimize
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
                # -- if levels aren't plotting check xl and xr and make sure they are making actual vectors.
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
        plt.savefig(f"{self.fig_dir}/{self.molecule.method}_adiabatplot_{self.OHDVR}OH{self.OODVR}OO.png")
        plt.close()

class AA2Dplots:
    def __init__(self, moleculeObj=None, TwoDnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.logData = moleculeObj.logData
        import os
        self.fig_dir = os.path.join(moleculeObj.mol_dir, "figures")
        self.TwoDResults = np.load(TwoDnpz)

    def plotProjections(self):
        grid = self.TwoDResults["grid"].squeeze()
        XHgrid = grid[0, :, 1]
        wfns = self.TwoDResults["wfns_array"]
        wfn_grids = wfns.reshape((len(wfns), grid.shape[0], grid.shape[1]))
        new_wfn = []
        for wfn_grid in wfn_grids:
            # for every XH all the OO values
            new_wfn.append(np.array([np.dot(wf_slice, wf_slice) for wf_slice in wfn_grid.T]))
        # plot each transition with ground state wfn and savefig
        # colors = ["royalblue", "darkmagenta", "mediumslateblue", "mediumblue"]
        colors = ["grey", "red", "orange", "deeppink"]
        for i in range(1, len(new_wfn)):
            plt.plot(XHgrid, new_wfn[0], "--", color=colors[0], label=r"$\Psi_{0,0}$")
            plt.plot(XHgrid, new_wfn[i], color=colors[i], label=r"$\Psi_{1, %d}$" % (i-1))
            plt.xlabel("$\mathrm{r_{XH}}$ ($\mathrm{\AA}$)", size=16)
            plt.ylabel("Probability Amplitude", size=16)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_OHwfnProjection_1{i-1}")
            plt.close()

class TMplots:
    def __init__(self, moleculeObj=None, OHDVRnpz=None, OODVRnpz=None, TwoDnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.TwoDnpz = TwoDnpz
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
            self._tmObj = TransitionMoment(moleculeObj=self.molecule, dimension="1D", TwoDnpz=self.TwoDnpz,
                                           OHDVRnpz=self.OHDVRnpz, OODVRnpz=self.OODVRnpz)
        return self._tmObj

    def TransitionMoments(self, color=None, ylim=(-0.2, 1)):
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
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_polyTDM.png")
        plt.close()

    def InterpolatedDips(self):
        """will plot dipoles (dots) and interpolation results (lines) for checking"""
        from McUtils.Plots import GraphicsGrid
        dip_struct = self.tmObj.make1D_DipStruct()  # this is a dictionary: (oo, xh, x, y, z) in each expansion
        expanTypes = self.tmObj.twoDexpanTypes
        color = [["k", "navy", "darkblue", "blue", "royalblue", "cornflowerblue", "lightskyblue", "lightsteelblue"],
                 ["k", "chocolate", "darkorange", "orange", "goldenrod", "gold", "wheat", "moccasin"],
                 ["k", "darkgreen", "green", "forestgreen", "seagreen", "limegreen", "lime", "mediumseagreen"]]
        comp = ["X", "Y", "Z"]
        plt.rcParams.update({'font.size': 16})
        val = dip_struct["dipSurf"][:, 0, 0]
        for k, oo in enumerate(val):  # loop through cuts
            main = GraphicsGrid(ncols=3, nrows=1)
            main.image_size = (1000, 400)
            for i, t in enumerate(expanTypes):  # loop through expansion types
                dip_vals = dip_struct[t]
                dip_vecs = dip_vals[k, :, 2:]
                xhs = dip_vals[k, :, 1]
                for j in np.arange(3):  # loop through x, y, z
                    # main[0, j].plot(dip_struct[k, :, 1], dip_struct[k, :, j + 2], "o",
                    #                 label=f"{comp[j]} Dipole", color=color[j][0])
                    main[0, j].plot_label = f"{comp[j]} Dipole"
                    main[0, j].plot(xhs, dip_vecs[:, j],
                                    label=f"{t} fit", color=color[j][i + 1], linewidth=4.0)
                    main[0, j].legend()
                    main[0, j].set_xlim(-0.25, 0.75)
                    main[0, j].set_ylim(-5, 10)
            main.figure.suptitle(f"Roo = {oo}")
            main.show()

    def componentTMs(self, ylim=None, xlim=None):
        comp = ["X", "Y", "Z"]
        x = Constants.convert(self.tmObj.mus[0]["dipSurf"][:, 0, 0], "angstroms", to_AU=False)
        mus = self.tmObj.mus[1]["dipSurf"]
        bigGrid = Constants.convert(self.tmObj.tdms[0], "angstroms", to_AU=False)
        exMus = self.tmObj.tdms[1]
        labelNames = {"dipSurf": "Dipole Surface", "quadOH": "Quadratic (XH) Dipole", "linOH": "Linear (XH) Dipole"}
        # colors = ["darkmagenta", "mediumslateblue", "mediumblue"]
        colors = ["green", "blue", "purple"]
        # fig = plt.figure(figsize=(6, 4), dpi=600)
        for j, v in enumerate(comp):
            for i, t in enumerate(labelNames.keys()):
                plt.plot(bigGrid, exMus[t][:, j], color=colors[i], label=labelNames[t], linewidth=2.0)
            for l, pt in enumerate(x):
                if pt == 2.3296 or pt == 2.5696 or pt == 2.8096:
                    plt.plot(pt, mus[l, j], "o", color="red", fillstyle="none")
                else:
                    plt.plot(pt, mus[l, j], 'ok')
            if ylim is not None:
                plt.ylim(*ylim)
            if xlim is not None:
                plt.xlim(*xlim)
            plt.title(f"{v} Component TDM", size=18)
            plt.legend(fontsize=14)
            plt.xlabel("$\mathrm{R_{OO}}$ ($\mathrm{\AA}$)", size=16)
            plt.ylabel("Transition Dipole Moment (Debye)", size=16)
            plt.tight_layout()
            # plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_{v}componentTDM_red.png",
            #             dpi=fig.dpi, bbox_inches="tight")
            plt.show()

class TM2Dplots:
    def __init__(self, moleculeObj=None, TwoDnpz=None, **kwargs):
        self.molecule = moleculeObj
        if self.molecule is None:
            raise Exception("No molecule to test")
        self.TwoDnpz = TwoDnpz
        self.TwoDResults = np.load(TwoDnpz)
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
        from McUtils.Plots import Graphics, GraphicsGrid, ListContourPlot
        Styled = Graphics.modified
        grid = self.tmObj.TwoDDips[0]
        dips = self.tmObj.TwoDDips[1]
        mini = np.amin(dips)
        maxi = np.amax(dips)
        comp = ['X', 'Y', 'Z']
        main = GraphicsGrid(ncols=3, nrows=1)
        main.image_size = (1800, 600)
        main.padding = ((0.05, 0.05), (0.1, 0.1))
        for i in np.arange(dips.shape[1]):
            opts = dict(
                plot_style=dict(cmap="viridis_r", levels=10, vmin=mini, vmax=maxi),
                figure=main[0, i],
                axes_labels=[Styled('$\mathrm{R_{OO}}$ ($\mathrm{\AA}$)', size=16),
                             Styled('$\mathrm{r_{XH}}$ ($\mathrm{\AA}$)', size=16)])
            main[0, i] = ListContourPlot(np.column_stack((grid[:, 0], grid[:, 1], dips[:, i])), **opts)
            main[0, i].plot_label = Styled(f'{comp[i]}-Component of Dipole', size=18)
        main.colorbar = {"graphics": main[0, 0].graphics}
        plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_2D_dipoleplots.png")
        plt.close()

    def componentDMs(self):
        from McUtils.Plots import Graphics, GraphicsGrid, ListContourPlot
        Styled = Graphics.modified
        comp = ["X", "Y", "Z"]
        labelNames = {"dipSurf": "Dipole Surface", "quadOH": "Quadratic (XH) Dipole", "linOH": "Linear (XH) Dipole"}
        Grid = Constants.convert(self.tmObj.TwoD_dms[0], "angstroms", to_AU=False)
        exMus = self.tmObj.TwoD_dms[1]
        mini = np.amin(exMus["dipSurf"])
        maxi = np.amax(exMus["dipSurf"])
        for i in np.arange(3):  # make one figure per comp
            main = GraphicsGrid(ncols=3, nrows=1)
            main.image_size = (1800, 600)
            main.padding = ((0.05, 0.05), (0.1, 0.1))
            opts = dict(
                plot_style=dict(cmap="viridis_r", levels=10, vmin=mini, vmax=maxi),
                axes_labels=[Styled('$\mathrm{R_{OO}}$ ($\mathrm{\AA}$)', size=16),
                             Styled('$\mathrm{r_{XH}}$ ($\mathrm{\AA}$)', size=16)])
            for j, k in enumerate(labelNames.keys()):
                main[0, j] = ListContourPlot(np.column_stack((Grid[:, 0], Grid[:, 1], exMus[k][:, i])),
                                             figure=main[0, j], **opts)
                main[0, j].plot_label = Styled(labelNames[k], size=18)
            main.figure.suptitle(f"{comp[i]} - Component", size=18)
            main.colorbar = {"graphics": main[0, 0].graphics}
            plt.tick_params(axis='both', which='minor', labelsize=14)
            plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{self.molecule.method}_2D_{comp[i]}_DMexpansions.png")
            plt.close()

    def plotDMcut(self, ylim=None, xlim=None):
        comp = ["X", "Y", "Z"]
        labelNames = {"dipSurf": "Dipole Surface", "quadOH": "Quadratic (XH) Dipole", "linOH": "Linear (XH) Dipole"}
        exMus = self.tmObj.TwoD_dms[1]
        Grid = self.TwoDResults["grid"].squeeze()
        Pot = self.TwoDResults["potential"]
        squarepot = np.reshape(Pot, (Grid.shape[0], Grid.shape[1]))
        mini = np.unravel_index(np.argmin(squarepot, axis=None), squarepot.shape)
        XHgrid = Grid[0, :, 1]
        # colors = ["darkmagenta", "mediumslateblue", "mediumblue"]
        colors = ["green", "blue", "purple"]
        for i, c in enumerate(comp):  # loop through components
            mu_cuts = []
            for t in labelNames.keys():  # loop through expansions
                # for every XH all the OO values
                mu_grid = exMus[t][:, i].reshape((Grid.shape[0], Grid.shape[1]))
                mu_cuts.append(mu_grid[mini[0], :])
            for j, k in enumerate(labelNames.keys()):
                plt.plot(XHgrid, mu_cuts[j], color=colors[j], label=labelNames[k])
            plt.xlabel("$\mathrm{r_{XH}}$ ($\mathrm{\AA}$)", size=16)
            plt.ylabel("Dipole Moment (Debye)", size=16)
            plt.title(f"{c}-component", size=18)
            if ylim is not None:
                plt.ylim(*ylim)
            if xlim is not None:
                plt.xlim(*xlim)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.show()
            # plt.savefig(f"{self.fig_dir}/{self.molecule.MoleculeName}_{c}_DipoleSurfCuts.png")
            plt.close()


