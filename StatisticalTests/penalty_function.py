import csv
import matplotlib


import matplotlib.pyplot as plt


def f_p(x, gamma, number_of_features):
    return ((gamma ** ((x - 1) / (number_of_features - 1))) - 1) / (gamma - 1)


N = 100
scale_pen = 0.25

x_values = range(1, N + 1)
y_values = map(lambda x: f_p(float(x), 10, N), x_values)

print(y_values)

fig, ax = plt.subplots()

ax.plot(map(lambda x: f_p(float(x), 5, N), range(1, N + 1)), 'k.', label=r'$\gamma=5$')

ax.plot(x_values, y_values, 'k-', label=r'$\gamma=10$')

plt.ylabel(r'$f_p(s)$')
plt.xlabel(r'Number of features, $|s|$, when $N = 100$')

ax.plot(x_values, map(lambda x: x * scale_pen, y_values), 'k--', label=r'$\gamma=10, k_p=0.25$')

ax.plot(map(lambda x: f_p(float(x), 100, N), range(1, N + 1)), 'k:', label=r'$\gamma=100$')



legend = ax.legend(loc='upper center', shadow=True, fontsize='small')

fig.savefig("out/penalty.png", pad_inches=1, dpi=250)
