import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(sys):
    udrive = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    mainD = os.path.join(udrive, "H+H2On Experimental Data")
    data_dict = dict()
    if sys == "tetramer":
        tet = np.loadtxt(os.path.join(mainD, "xyDataTetramer_allH.csv"), skiprows=1, delimiter=",")
        data_dict["dat"] = tet
        norm = tet[:, 1] / tet[np.argmax(tet[:, 1]), 1]
        data_dict["norm_dat"] = np.column_stack((tet[:, 0], norm))
        data_dict["xlim"] = (2300, 4000)
        data_dict["OHfreq"] = 2650
        data_dict["combfreq"] = 3017
    elif sys == "trimer":
        tri = np.loadtxt(os.path.join(mainD, "xyDataTrimer_1He_lower.csv"),  skiprows=1, delimiter=",")
        data_dict["dat"] = tri
        norm = tri[:, 1] / tri[np.argmax(tri[:, 1]), 1]
        data_dict["norm_dat"] = np.column_stack((tri[:, 0], norm))
        data_dict["xlim"] = (1000, 3000)
    else:
        raise Exception(f"No experimental data for {sys}")
    return data_dict

def calc_ratio(sys):
    dat_dict = load_data(sys)
    x_vals = dat_dict["dat"][:, 0]
    y_vals = dat_dict["dat"][:, 1]
    OH_inds = np.argwhere(np.logical_and(2400 < x_vals, x_vals < 2900))
    comb_inds = np.argwhere(np.logical_and(3000 < x_vals, x_vals < 3200))
    OH_maxi = np.argmax(y_vals[OH_inds])+OH_inds[0]
    OH_intensity = x_vals[OH_maxi] * y_vals[OH_maxi]
    print("F: ", x_vals[OH_maxi], "ME: ", y_vals[OH_maxi], "I: ", OH_intensity)
    comb_maxi = np.argmax(y_vals[comb_inds])+comb_inds[0]
    comb_intensity = x_vals[comb_maxi] * y_vals[comb_maxi]
    print("F: ", x_vals[comb_maxi], "ME: ", y_vals[comb_maxi], "I: ", comb_intensity)
    ratio = comb_intensity / OH_intensity * 100
    print("percent", ratio)


if __name__ == '__main__':
    calc_ratio("tetramer")
