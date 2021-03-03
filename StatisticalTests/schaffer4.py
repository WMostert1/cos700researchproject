# rastrigin_graph.py


import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-100.0, 100.0, 100)
Y = np.linspace(-100.0, 100.0, 100)
X, Y = np.meshgrid(X, Y)

Z = 0.5 + ((np.cos(np.sin(np.abs(X ** 2 - Y ** 2)))) ** 2 - 0.5) / (1 + 0.001 * (X ** 2 + Y ** 2)) ** 2

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap=cm.nipy_spectral, linewidth=0.08,
                antialiased=True)
plt.savefig('out/schaffern4_graph.png', pad_inches=1, dpi=150, transparent=True)
