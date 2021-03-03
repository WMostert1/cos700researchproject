# Friedman test
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare
import csv

populations = []
with open('/Users/wmostert/Development/cos700researchproject/out/bfiTable.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    line_count = 0
    for row in csv_reader:
        if line_count > 2:
            result = map(lambda x: float(x.replace(",", ".")), row[1:-1])
            populations.append(result)
        line_count += 1

stat, p = friedmanchisquare(*zip(*populations))
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
