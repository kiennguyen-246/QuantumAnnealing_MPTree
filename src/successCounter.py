from steinerTree import *

trials = 1
numReads = 1000
totalSuccessCount = 0
totalOptimalCount = 0
# Graph input. Must be vertices from 0 to n - 1
k = 0
terminals = []
with open("steiner.inp", "r") as inputFile:
    ln = inputFile.readline()
    n, m = map(int, ln.split())
    g = nx.DiGraph()
    for i in range(0, m):
        ln = inputFile.readline()
        u, v, w = map(int, ln.split())
        g.add_edge(u, v, weight=w)
        g.add_edge(v, u, weight=w)
    k = int(inputFile.readline())
    ln = inputFile.readline()
    terminals = list(map(int, ln.split()))
for i in range(0, trials):
    print("Attempt #{}".format(i + 1))
    ans = fowler(g=g,
                terminals=terminals,
                __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1,
                numReads=numReads,
                chainStrengthPrefactor=0.3,
                annealing_time=200)
    totalSuccessCount += ans["success_rate"]
    totalOptimalCount += ans["optimal_rate"]

print("\n\n\n------------------------")
print("Steiner tree creation rate: {}/{}".format(totalSuccessCount, trials * numReads))
print("Optimal rate: {}/{}".format(totalOptimalCount, trials * numReads))
