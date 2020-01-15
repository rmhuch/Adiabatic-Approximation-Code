import os
import numpy as np

def find_FancyF(FD_file, woo=None, woh=None):
    from Converter import Constants
    from McUtils.Zachary import finite_difference
    mO = Constants.mass("O", to_AU=True)
    mH = Constants.mass("H", to_AU=True)
    freqoo = Constants.convert(woo, "wavenumbers", to_AU=True)
    muOO = mO / 2
    freqoh = Constants.convert(woh, "wavenumbers", to_AU=True)
    muOH = ((2 * mO) * mH) / ((2 * mO) + mH)
    finite_vals = np.loadtxt(FD_file, skiprows=7)
    finite_vals = finite_vals[:, 2:]
    finite_vals[:, 0] *= 2
    ens = finite_vals[:, 2]
    print("Energy Difference from Minimum: ", Constants.convert(finite_vals[:, 2]-min(finite_vals[:, 2]), "wavenumbers", to_AU=False))
    ohDiffs = np.array([ens[1]-ens[0], ens[6]-ens[5], ens[11]-ens[10], ens[16]-ens[15], ens[21]-ens[20]])
    print("OH energy differences: ", Constants.convert(ohDiffs, "wavenumbers", to_AU=False))
    finite_vals[:, :2] = Constants.convert(finite_vals[:, :2], "angstroms", to_AU=True)  # convert to bohr for math
    idx = np.lexsort((finite_vals[:, 0], finite_vals[:, 1]))  # resort so same oh different oo
    finite_vals = finite_vals[idx]
    finite_vals = finite_vals.reshape((5, 5, 3))
    FR = np.zeros(5)
    # compute first derivative wrt oo FR
    for j in range(finite_vals.shape[0]):
        x = finite_vals[j, :, 0]  # roos
        y = finite_vals[j, :, 2]  # energies
        print("OO energy difference: ", Constants.convert((y[1]-y[0]), "wavenumbers", to_AU=False))
        FR[j] = finite_difference(x, y, 1, end_point_precision=0, stencil=5, only_center=True)[0]
    print(f"FR: {FR}")
    # compute mixed derivative FrrR
    FrrR = finite_difference(finite_vals[:, 1, 1], FR, 2, end_point_precision=0, stencil=5, only_center=True)[0]
    print(f"FrrR: {FrrR}")
    Qoo = np.sqrt(1 / muOO / freqoo)
    Qoh = np.sqrt(1 / muOH / freqoh)
    fancyF = FrrR * Qoh ** 2 * Qoo
    return Constants.convert(fancyF, "wavenumbers", to_AU=False)

# udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# dir = os.path.join(udrive, "H9O4pls")
file1 = "2D_finiteSPEtri_01.dat"
file2 = "2D_finiteSPEtri_01_005.dat"
file3 = "2D_finiteSPEtri_005.dat"
file4 = "2D_finiteSPEtri_008.dat"
file5 = "2D_finiteSPEtri_01_008.dat"
omegaOO = 421.37
omegaOH = 2508.70

print("delta = 0.01")
test1 = find_FancyF(file1, woo=omegaOO, woh=omegaOH)
print(f"Fancy F: {test1}")
print("delta = 0.01 / 0.005")
test2 = find_FancyF(file2, woo=omegaOO, woh=omegaOH)
print(f"Fancy F: {test2}")
print("delta = 0.005")
test3 = find_FancyF(file3, woo=omegaOO, woh=omegaOH)
print(f"Fancy F: {test3}")
print("delta = 0.008")
test4 = find_FancyF(file4, woo=omegaOO, woh=omegaOH)
print(f"Fancy F: {test4}")
print("delta = 0.01 / 0.008")
test5 = find_FancyF(file5, woo=omegaOO, woh=omegaOH)
print(f"Fancy F: {test5}")

