# rastrigin_graph.py


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-5.12, 5.12, 100)
Y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(X, Y)

Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
    (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20

fig, ax = plt.subplots()

ax.tricontour(X, Y, Z,  levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(X, Y, Z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr2, ax=ax)

plt.savefig('out/rastrigin_graph.png', pad_inches=1, dpi=150, transparent=True)
