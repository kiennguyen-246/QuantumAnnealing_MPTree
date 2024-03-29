from steinerTree import * 

trials = 1
numReads = 1000
totalSuccessCount = 0
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
        g.add_edge(u, v, weight = w)
        g.add_edge(v, u, weight = w)
    k = int(inputFile.readline())
    ln = inputFile.readline()
    terminals = list(map(int, ln.split()))
for i in range(0, trials):
    print("Attempt #{}".format(i + 1))
    totalSuccessCount += steinerTreeSuccessCount(g = g,
                                                 terminals = terminals,
                                                 __lambda = max([g[u][v]['weight'] for (u, v) in list(g.edges)]) + 5,
                                                 numReads = numReads,
                                                 chainStrengthPrefactor = 0.3,
                                                 annealing_time = 350)

print("\n\n\n------------------------")
print("Total success rate: {}/{}".format(totalSuccessCount, trials * numReads))