import numpy as np
import matplotlib.pyplot as plt

# tet = np.loadtxt("xyDataTetramer_1He.csv", skiprows=1, delimiter=",")
# tri = np.loadtxt("xyDataTrimer_1He_lower.csv",  skiprows=1, delimiter=",")
# triu = np.loadtxt("xyDataTrimer_1He_upper.csv",  skiprows=1, delimiter=",")

tetallH = np.loadtxt("xyDataTetramer_allH.csv", skiprows=1, delimiter=",")
triallH = np.loadtxt("xyDataTrimer_allH.csv", skiprows=1, delimiter=",")
# tetallHnorm = tetallH[:, 1] / np.argmax(tetallH[:, 1])
plt.plot(tetallH[:, 0], tetallH[:, 1], "-r", linewidth=3.0)
plt.xlim(2400, 3600)
plt.show()
