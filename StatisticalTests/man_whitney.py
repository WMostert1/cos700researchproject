# coding=utf-8
import csv
import os
import matplotlib
import statistics
import numpy as np

matplotlib.use('TkAgg')
import statistics
from matplotlib.pyplot import boxplot
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

run_identifier = "2021-03-01T18:41:08.947Z|kc(1.0)|kp(0.25)"

if not os.path.exists("out/" + run_identifier):
    os.makedirs("out/" + run_identifier)

directory_root = '/Users/wmostert/Development/cos700researchproject/out/' + run_identifier + "/fitness/"
# data -> [ ( data_set_name, [ (algorithm_name, [ fitness_values ] ) ] ) ]
data = []

result_table = []


def get_value(arr, val):
    for a in arr:
        if val == a[0] or val == a[1]:
            return a
    return None


def read_data(path):
    populations = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count > 1:
                populations.append(float(row[0].replace(",", ".")))
            line_count += 1
    return populations


# [ (data_set_name , [ ( algorithm_name, bfi_value ] ) ]
bfi_data = []
algorithm_names = []
algorithm_bfi_data = [[] for i in range(0, 6)]
full_algorithm_names = ['AMSO', 'GAFS', 'SBFS', 'SFFS', 'PCFS', 'IGFS']

# Read BFI values from csv file but exclude random feature selection
with open('/Users/wmostert/Development/cos700researchproject/out/' + run_identifier + '/bfiTable.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            algorithm_names = row[2:-1]

        if line_count > 1:
            data_set_name = row[0].replace(".arff", "")
            bfi_values = map(lambda x: float(x.replace(",", ".")), row[2:-1])
            annotated_bfi_values = []
            for i in range(0, len(bfi_values)):
                annotated_bfi_values.append([algorithm_names[i], bfi_values[i]])
                algorithm_bfi_data[i].append(bfi_values[i])
            bfi_data.append((data_set_name, annotated_bfi_values))
        line_count += 1

full_dataset_names = []

with open('/Users/wmostert/Development/cos700researchproject/out/' + run_identifier + '/dataSetInformation.tex',
          'r') as dataset_reader:
    line = dataset_reader.readline()
    while line != '':  # The EOF char is an empty string
        if ".arff" in line:
            full_dataset_names.append(line[line.index("&") + 1:line.index(".arff") + 5])

        line = dataset_reader.readline()

algorithm_ranks = []



# This is used to mark algorithms as stochastic based on the BFI table name excluding the first one which is random
stochastic_indexes = ["AMSO", "GA"]

bfi_median_latex_table_data = "";
ds_index = 0
for ds in bfi_data:
    print(ds)
    # Replace the means of the stochastic algorithms with median and IQR
    row = ["D"+str(ds_index+1)]

    samples = read_data(directory_root + full_dataset_names[ds_index] + "/Random.csv")
    q3, q1 = np.percentile(np.array(samples), [75, 25])
    iqr = q3 - q1
    median = statistics.median(samples)
    row.append('{:.4f}'.format(median) + "(" + '{:.2f}'.format(q3)+"-"+'{:.2f}'.format(q1)+")")

    for algorithm in ds[1]:
        if algorithm[0] in stochastic_indexes:
            samples = read_data(directory_root + full_dataset_names[ds_index] + '/' + algorithm[0] + ".csv")
            q3, q1 = np.percentile(np.array(samples), [75, 25])
            iqr = q3 - q1
            median = statistics.median(samples)
            row.append('{:.4f}'.format(median) + "(" + '{:.2f}'.format(q3)+"-"+'{:.2f}'.format(q1)+")")
            algorithm[1] = median
        else:
            row.append('{:.4f}'.format(algorithm[1]))

    bfi_median_latex_table_data += "&".join(row)+"\\\\\n"
    ds_index += 1

print("BFI Median Data")
print(bfi_median_latex_table_data)

# Build box plots of the BFI values per algorithm

# temp reordering

# bsfs = algorithm_bfi_data[5]
# algorithm_bfi_data[5] = algorithm_bfi_data[2]
# algorithm_bfi_data[2] = bsfs

# Multiple box plots on one Axes
fig, ax = plt.subplots()
bp = ax.boxplot(algorithm_bfi_data)

# ax.set_xticklabels(['AMSO', 'GAFS', 'BSFS', 'FSFS', 'PCFS', 'IGFS'])
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['medians'], color='black')
plt.setp(bp['fliers'], color='black')

fig.savefig("out/" + run_identifier + "/multi-box-bfi.png")

## Create box plots of just fitness

algorithm_fitness_data = [[] for i in range(0, 6)]


def all_vals_in_array_equal(arr):
    return all(elem == arr[0] for elem in arr)


# Read BFI values from csv file but exclude random feature selection
# with open('/Users/wmostert/Development/cos700researchproject/out/fitnessTable.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=';')
#     line_count = 0
#     for row in csv_reader:
#         if line_count > 0:
#             fitness_values = map(lambda x: float(x.replace(",", ".")), row[3:-1])
#             if not all_vals_in_array_equal(fitness_values):
#                 for i in range(0, len(fitness_values)):
#                     algorithm_fitness_data[i].append(fitness_values[i])
#             else:
#                 print("Skip "+row[0])
#         line_count += 1
#
# fig, ax = plt.subplots()
# bp = ax.boxplot(algorithm_fitness_data)
# plt.setp(bp['boxes'], color='black')
# plt.setp(bp['whiskers'], color='black')
# plt.setp(bp['medians'], color='black')
# plt.setp(bp['fliers'], color='black')
#
# fig.savefig("out/"+run_identifier+"multi-box-fitness.png")


for root, dirs, files in os.walk(directory_root):
    for dataset in dirs:
        if ".arff" in str(dataset):
            path = directory_root + dataset
            dataset_data = []
            for algorithm in [x[2] for x in os.walk(path)][0]:
                if 'Random' not in algorithm:
                    values = read_data(path + '/' + algorithm)
                    # Add a tuple (algorithm_name_from_file, fitness_values_recorded)
                    dataset_data.append((algorithm.replace(".csv", ""), values))
            # Add the list of algorithm, fitness tuples to data as a tuple ()
            data.append((dataset.replace(".arff", ""), dataset_data))
# Mann-Whitney U test


shared_ranks = []

data_set_index = 1
for problem_data in data:
    data_set_name = problem_data[0]
    print ('--------' + data_set_name + '--------')
    number_of_algorithms = len(problem_data[1])
    equal_algorithms = ("D" + str(data_set_index), [])
    # For all pairwise comparisons of stochastic algorithms
    for i in range(number_of_algorithms):
        # Do a Mann Whitney U test for all algorithms that are stochastic
        for j in range(i, number_of_algorithms):
            if i == j:
                continue

            algorithm_one_name = problem_data[1][i][0]
            algorithm_two_name = problem_data[1][j][0]
            print (algorithm_one_name + " vs " + algorithm_two_name)
            try:
                stat, p = mannwhitneyu(problem_data[1][i][1], problem_data[1][j][1], alternative="two-sided")

                print('Statistics=%.3f, p=%.3f' % (stat, p))
                # interpret
                alpha = 0.05
                if p > alpha:
                    print('Same distribution (fail to reject H0)')
                    equal_algorithms[1].append((algorithm_one_name, algorithm_two_name))
                else:
                    print('Different distribution (reject H0)')

            except ValueError as e:
                if e.message == 'All numbers are identical in mannwhitneyu':
                    print('Same distribution (fail to reject H0) - ALL SAME VALUES')
                    equal_algorithms[1].append((algorithm_one_name, algorithm_two_name))
                else:
                    raise e

        # do interval checking for deterministic algorithms
        stdeviation = statistics.stdev(problem_data[1][i][1])
        q3, q1 = np.percentile(np.array(problem_data[1][i][1]), [75, 25])
        iqr = q3 - q1
        mean = statistics.mean(problem_data[1][i][1])
        median = statistics.median(problem_data[1][i][1])

        for d in range(len((bfi_data[data_set_index - 1][1]))):
            algorithm_j = bfi_data[data_set_index - 1][1][d][0]
            if algorithm_j in stochastic_indexes:
                continue

            bfi_val = bfi_data[data_set_index - 1][1][d][1]
            if q3 >= bfi_val >= q1:
                # if bfi_val <= mean + stdeviation and bfi_val >= mean - stdeviation:
                equal_algorithms[1].append((problem_data[1][i][0], algorithm_j))
    data_set_index += 1
    shared_ranks.append(equal_algorithms)



print ("-----------")

# build latex table for Mann-Whitney U test

shared_ranks_table = ("\\begin{table}\n"
                      "\\caption{Mann-Whitney U Results}\n"
                      "\\label{tbl:mann:whitney:u}\n"
                      "\\begin{tabular}{ll}\n"
                      "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
                      "Data Set & Algorithms \\\\\n"
                      "\\noalign{\\smallskip}\\hline\n"
                      )

for r in shared_ranks:
    if len(r[1]) == 0:
        continue
    shared_ranks_table += r[0].replace(",arff", "") + "&" + ", ".join(
        map(lambda v: str(v[0]) + "-" + str(v[1]) + "\\\\\n", r[1]))
shared_ranks_table += ("\\noalign{\\smallskip}\\hline\n"
                       "\\end{tabular}\n"
                       "\\end{table}\n")

with open("out/" + run_identifier + "/shared_ranks_table.tex", "w") as text_file:
    text_file.write(shared_ranks_table)

# [ (dataset, [ranks])]
ranked_bfi_data = []

# sort the BFI values so that higher values are first
for di in range(0, len(bfi_data)):
    bfi_data[di][1].sort(key=lambda x: x[1], reverse=True)

data_set_index = 0
i = 0
for bfi in bfi_data:
    i += 1
    data_set_name = "D" + str(i)
    algorithm_values = bfi[1]

    ranks_result = []
    for x in range(0, len(algorithm_names)):
        ranks_result.append(0)

    previous_used_rank = 1

    algorithms_that_share_rank = shared_ranks[data_set_index]

    # Check determistic algorithms and add their shared ranks if they are the same
    for alg_i in range(0,len(algorithm_values)):
        for alg_j in range(0, len(algorithm_values)):
            if alg_i == alg_j:
                continue
            if algorithm_values[alg_i][0] in stochastic_indexes or algorithm_values[alg_j][0] in stochastic_indexes:
                continue
            if algorithm_values[alg_i][1] == algorithm_values[alg_j][1]:
                algorithms_that_share_rank[1].append((algorithm_values[alg_i][0], algorithm_values[alg_j][0]))

    for algorhtm_index in range(0, len(algorithm_values)):
        algorithm_name = algorithm_values[algorhtm_index][0]
        algorithm_names_index = algorithm_names.index(algorithm_name)
        if ranks_result[algorithm_names_index] == 0:
            ranks_result[algorithm_names_index] = previous_used_rank
        # shared_ranks_indexes_used = []
        for shared_ranks_i in range(0, len(algorithms_that_share_rank[1])):
            if algorithm_name in algorithms_that_share_rank[1][shared_ranks_i]:
                # if shared_ranks_i in shared_ranks_indexes_used:
                #     continue

                # shared_ranks_indexes_used.append(shared_ranks_i)
                alg_one_index = algorithm_names.index(algorithms_that_share_rank[1][shared_ranks_i][0])
                alg_two_index = algorithm_names.index(algorithms_that_share_rank[1][shared_ranks_i][1])
                if ranks_result[alg_one_index] == 0:
                    ranks_result[alg_one_index] = previous_used_rank
                if ranks_result[alg_two_index] == 0:
                    ranks_result[alg_two_index] = previous_used_rank
        if previous_used_rank in ranks_result:
            previous_used_rank += 1


    # blacklist = []
    #
    # previous_fitness = None
    # while len(algorithm_values) != 0:
    #     algorithm_name = algorithm_values[0][0]
    #
    #     if algorithm_name in blacklist:
    #         # Algorithms that rank together have already been ranked
    #         del algorithm_values[0]
    #         continue
    #
    #     algorithm_names_index = algorithm_names.index(algorithm_name)
    #
    #     # The rank is the index of the ranked bfi data
    #     rank = previous_used_rank
    #
    #     # If there is a shared rank then find both
    #
    #     shared_algorithm = get_value(algorithms_that_share_rank[1], algorithm_name)
    #     if shared_algorithm is not None:
    #         algorithm_names_index_one = algorithm_names.index(shared_algorithm[0])
    #         algorithm_names_index_two = algorithm_names.index(shared_algorithm[1])
    #
    #         blacklist.append(shared_algorithm[0])
    #         blacklist.append(shared_algorithm[1])
    #
    #         if previous_fitness != algorithm_values[0][1] and previous_fitness is not None:
    #             previous_used_rank += 1
    #             if previous_fitness is None:
    #                 previous_used_rank -= 1
    #             rank = previous_used_rank
    #         previous_fitness = algorithm_values[0][1]
    #
    #         ranks_result[algorithm_names_index_one] = rank
    #         ranks_result[algorithm_names_index_two] = rank
    #
    #         del algorithm_values[0]
    #         continue
    #
    #     if previous_fitness != algorithm_values[0][1]:
    #         previous_used_rank += 1
    #         if previous_fitness is None:
    #             previous_used_rank -= 1
    #         rank = previous_used_rank
    #         # if rank == 1 and i == 1:
    #         #     rank = previous_used_rank
    #     previous_fitness = algorithm_values[0][1]
    #
    #     ranks_result[algorithm_names_index] = rank
    #     del algorithm_values[0]

    ranked_bfi_data.append((data_set_name, ranks_result))
    data_set_index += 1

print ("-----------")

# build latext table

ranks_table = ("\\begin{table}\n"
               "\\caption{Algorithm Performance Rankings}\n"
               "\\label{tbl:algorithm:rank}\n"
               "\\begin{tabular}{lllllll}\n"
               "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n")
italic_names = map(lambda a: full_algorithm_names[a], range(0, len(algorithm_names)))
ranks_table += "&" + "&".join(italic_names) + "\\\\\n\\noalign{\\smallskip}\\hline\n"
data_set_index = 1
for r in ranked_bfi_data:
    # ranks_table += r[0].replace(",arff", "") + "&" + "&".join(map(lambda v: str(v), r[1])) + "\\\\\n"
    ranks_table += "D" + str(data_set_index) + "&" + "&".join(map(lambda v: str(v), r[1])) + "\\\\\n"
    data_set_index += 1
ranks_table += ("\\noalign{\\smallskip}\\hline\n"
                "\\end{tabular}\n"
                "\\end{table}\n")

with open("out/" + run_identifier + "/ranks_table.tex", "w") as text_file:
    text_file.write(ranks_table)

for r in ranked_bfi_data:
    print(r)

# determine the number of times each algorithm was the worst and the best
best_list = [0 for x in range(0, len(algorithm_names))]
worst_list = [0 for x in range(0, len(algorithm_names))]

for r in ranked_bfi_data:
    ranks = r[1]
    if all(elem == 1 for elem in ranks):
        continue
    for i in range(0, len(algorithm_names)):
        if ranks[i] == min(ranks):
            best_list[i] += 1
        if ranks[i] == max(ranks):
            worst_list[i] += 1

# build instance count table for best and worst
rank_count_table = ("\\begin{table}\n"
                    "\\caption{Best and Worst Algorithm Counts}\n"
                    "\\label{tbl:best:worst:count}\n"
                    "\\begin{tabular}{lll}\n"
                    "\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}\n"
                    "Algorithm&Best&Worst\\\\\n"
                    "\\noalign{\\smallskip}\\hline\n")
for i in range(0, len(algorithm_names)):
    rank_count_table += full_algorithm_names[i] + "&" + str(best_list[i]) + " (" + str(
        int(100.0 * best_list[i] / sum(best_list))) + "\\%)&" + str(
        worst_list[i]) + " (" + str(int(100.0 * worst_list[i] / sum(worst_list))) + "\\%)\\\\\n"
rank_count_table += ("\\noalign{\\smallskip}\\hline\n"
                     "\\end{tabular}\n"
                     "\\end{table}\n")
# remove data sets where all algorithms are on the same rank

with open("out/" + run_identifier + "/rank_count_table.tex", "w") as text_file:
    text_file.write(rank_count_table)

algorithm_specific_ranks = []

for alg_i in range(0, len(algorithm_names)):
    alg_ranks = []
    for ds in ranked_bfi_data:
        if all(elem == 1 for elem in ds[1]):
            continue
        alg_ranks.append(ds[1][alg_i])
    algorithm_specific_ranks.append(alg_ranks)

# Calculate box plots
no_boxplot_rows = 2
# fig, axs = plt.subplots(no_boxplot_rows, len(algorithm_names) / no_boxplot_rows)

x = -1
# for index in range(0, len(algorithm_names)):
#
#     if index % (len(algorithm_names) / no_boxplot_rows) == 0:
#         x += 1
#
#     y = index % (len(algorithm_names) / no_boxplot_rows)
#     bp = axs[x, y].boxplot(algorithm_specific_ranks[index])
#     plt.setp(bp['boxes'], color='black')
#     plt.setp(bp['whiskers'], color='black')
#     plt.setp(bp['medians'], color='black')
#     plt.setp(bp['fliers'], color='black')
#
#     # axs[x, y].set_title(algorithm_names[index], fontsize=10, y=1.08)
#     axs[x, y].set_title("A" + str(index + 1), fontsize=10, y=1.08)
#     axs[x, y].set_ylim([1, len(algorithm_names)])
#     axs[x, y].get_xaxis().set_visible(False)
#     axs[x, y].set(frame_on=False)

# fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
bp = ax.boxplot(algorithm_specific_ranks)
ax.set_xticklabels(full_algorithm_names)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['medians'], color='black')
plt.setp(bp['fliers'], color='black')

fig.savefig("out/" + run_identifier + "/algorithm_rank_boxplots.png", dpi=300)
