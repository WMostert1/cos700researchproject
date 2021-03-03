import csv
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from statistics import stdev

legend = []
values = []
markers = ["o", "*", "^", "h", "s", "D"]

colors = ["b", "g", "r", "c", "m", "k"]

for i in range(6):
    values.append(([], []))

# Read BFI values from csv file but exclude random feature selection
# [attributes, instances, classes]
data_set_info = []

data_set_bfis = []

with open('/Users/wmostert/Development/cos700researchproject/out/dataSetInformation.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count > 0:
            data_set_info.append(map(lambda x: int(x), row[2:len(row)-1]))
        line_count += 1

with open('/Users/wmostert/Development/cos700researchproject/out/bfiTable.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # legend = row[2:-1]
            legend = map(lambda x: "A" + str(x), range(1, 7))

        if line_count > 1:
            bfi_values = map(lambda x: float(x.replace(",", ".")), row[2:-1])
            data_set_bfis.append(bfi_values)
            ##Change what is measured here
            measure = data_set_info[line_count - 2][1]
            for i in range(0, len(bfi_values)):
                print(str(i) + " - " + str(measure))
                x = measure
                y = bfi_values[i]
                values[i][0].append(x)
                values[i][1].append(y)

        line_count += 1

fig, ax = plt.subplots()
for i in range(6):
    x = values[i][0]
    y = values[i][1]
    ax.scatter(x, y, label=legend[i], marker=markers[i], c=colors[i])

ax.legend(fontsize="small", loc='lower center', bbox_to_anchor=(0.5, -.3), ncol=2)
ax.grid(True)
plt.xlabel("# Instances")
plt.ylabel("BFI(s)")
fig.subplots_adjust(bottom=0.2)
fig.savefig("out/scatter-instances.png", pad_inches=1, dpi=150)

std_dev_bfi_vals = [(max(bfis) - min(bfis)) for bfis in data_set_bfis]
stdev_x = [info[1] for info in data_set_info]

fig, ax = plt.subplots()

ax.scatter(stdev_x, std_dev_bfi_vals, c='k')
corr = np.corrcoef(stdev_x, std_dev_bfi_vals)
print("Correlation coefficient is : " + str(corr))


ax.grid(True)
plt.xlabel("# Instances")
plt.ylabel("Range")
# plt.xlim(0, 800)

fig.savefig("out/scatter-instances-range.png", pad_inches=1, dpi=150)







