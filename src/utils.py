from itertools import combinations
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def reportTable(reportFile, response):
    # reportFile.write("## Response\n")
    reportFile.write("|Sample|Energy|Occurrences|\n")
    reportFile.write("|---|---|---|\n")
    for i in range(0, len(response.record.sample)):
        reportFile.write("|`{}`|{}|{}|\n".format(
            response.record.sample[i], response.record.energy[i], response.record.num_occurrences[i]))


def add_qubo(q1=defaultdict, q2=defaultdict, size=0):
    """
    Add two QUBOs.
    """
    for i in range(0, size):
        for j in range(0, size):
            q1[(i, j)] += q2[(i, j)]
    return q1

def sum_max_1(q=defaultdict, size=0, x=0, y=0, z=0, __lambda=1):
    """
    Constraint: max(1, x + y) = z, quadratic form: H = x^2 + y^2 + z^2 + xy - 2xz - 2yz
    """
    q[(x, x)] += __lambda
    q[(y, y)] += __lambda
    q[(z, z)] += __lambda
    q[(x, y)] += __lambda
    q[(x, z)] -= 2 * __lambda
    q[(y, z)] -= 2 * __lambda
    return q

def mul(coef1=[], freeCoef1=0, coef2=[], freeCoef2=0, size=0, __lambda=1):
    """
    Multiply two linear terms.
    """
    q = defaultdict(int)
    for i in range(0, size):
        for j in range(0, size):
            q[(i, j)] += __lambda * coef1[i] * coef2[j]
    for i in range(0, size):
        q[(i, i)] += __lambda * coef1[i] * freeCoef2
    for i in range(0, size):
        q[(i, i)] += __lambda * freeCoef1 * coef2[i]
    return {
        "q": q,
        "offset": __lambda * freeCoef1 * freeCoef2
    }

def square(coef=[], freeCoef=0, size=0, __lambda=1):
    """
    Add a quadratic term to the QUBO.
    """
    q = defaultdict(int)
    for i in range(0, size):
        q[(i, i)] += __lambda * coef[i] ** 2
    for i in range(0, size):
        q[(i, i)] += __lambda * 2 * freeCoef * coef[i]
    for (i, j) in combinations(range(0, size), 2):
        q[(i, j)] += __lambda * 2 * coef[i] * coef[j]
    return {
        "q": q,
        "offset": __lambda * freeCoef ** 2
    }

def calculate(q=defaultdict(int), offset=0, x=[]):
    """
    Calculate the penalty.
    """
    h = 0
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            h += q[(i, j)] * x[i] * x[j]
    return h + offset

def plotBarChart(key, value, xLabel="", yLabel="", labelArr=[]):
    """
    Plot a bar chart.
    """
    plt.bar(key, value, label=labelArr)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()
