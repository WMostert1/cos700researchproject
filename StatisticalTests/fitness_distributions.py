import csv
import matplotlib
import math

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

data = []

with open('/Users/wmostert/Development/cos700researchproject/out/fitnessfrequency.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0

    i = -1
    data_set_name = ""
    sub_data = []
    sub_keys = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            continue

        if "arff" in row[0]:
            # Flush to data
            if i > -1:
                data.append((data_set_name, (sub_data, sub_keys)))
                sub_data = []
                sub_keys = []
            i += 1
            data_set_name = row[0].replace(",arff", "")
            continue

        sub_data.append(int(row[1]))
        sub_keys.append(row[0])
        line_count += 1

for d in data:
    print(d)

number_of_datasets = len(data)
number_of_rows = 16
columns = int(math.ceil(float(number_of_datasets) / number_of_rows))

print("Numbner of rows " + str(number_of_rows))
print("Numbner of cols " + str(columns))

fig, axs = plt.subplots(number_of_rows, columns)

print("Number of data_sets" + str(number_of_datasets))

x = -1
for index in range(0, len(data)):
    if index % columns == 0:
        x += 1
    y = index % columns

    print("i : " + str(index)+ ", x : " + str(x) + ", y : " + str(y))

    #print(str(len(data[index][1][1])) + "-" + str(len(data[index][1][0])))

    axs[x][y].bar(range(len(data[index][1][1])), data[index][1][0])

fig.savefig("out/fitness_distributions.png")

# N_points = 100000
#
# n_bins = 20
#
# # Generate a normal distribution, center at x=0 and y=5
# x = np.random.randn(N_points)
# y = .4 * x + np.random.randn(100000) + 5
#
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#
# # We can set the number of bins with the `bins` kwarg
# axs[0].hist(x, bins=n_bins)
# axs[1].hist(y, bins=n_bins)
