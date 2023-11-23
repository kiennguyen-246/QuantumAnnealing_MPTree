from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from dimod.binary import BinaryQuadraticModel
import dwave.inspector
from utils import *


def lucas(g, terminals,
          numReads=1000,
          __lambda=1,
          chainStrengthPrefactor=0.3,
          annealing_time=200):
    """
    https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full
    """
    # First initialization
    reportFile = open("steinerTree_response.md", "w")
    n = len(g.nodes)
    m = len(g.edges)
    edgeList = list(g.edges)
    udiEdgeList = list(filter(lambda x: x[0] < x[1], edgeList))

    def inEdgeList(u, v):
        return (u, v) in edgeList

    maxDepth = len(terminals) // 2 + 1

    # Hamiltonian parameters
    B = 1
    A = __lambda * B

    # Get the position of a variable in the QUBO
    def get(var="xv", depth=0, param1=0, param2=0):
        """ xv[i][x]: vertex x at depth i \\
            xe[i][x][y]: edge (x, y), y at depth i and x at depth i - 1 \\
            yv[x]: vertex x is in the tree \\
            ye[x][y]: edge (x, y) is in the tree"""
        if (var == "xv"):
            if (depth == 0):
                return param1
            return n + (depth - 1) * (n + m) + param1
        if (var == "xe"):
            if (inEdgeList(param1, param2) == False):
                return -1
            return n + (depth - 1) * (n + m) + n + edgeList.index((param1, param2))
        if (var == "yv"):
            return n + (maxDepth - 1) * (n + m) + param1
        if (var == "ye"):
            if (inEdgeList(param1, param2) == False):
                return -1
            return n + (maxDepth - 1) * (n + m) + n + udiEdgeList.index((param1, param2))

    # Generate QUBO
    q = defaultdict(int)
    qSize = (maxDepth + 1) * (n + m) - m - m // 2
    offset = 0

    def constraint1():
        """
        One vertex should be the root.
        """
        q = defaultdict(int)
        offset = 0
        coef2 = [0] * qSize
        for v in range(0, n):
            coef2[get(var="xv", depth=0, param1=v)] += 1
        q = addQubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["q"], size=qSize)
        offset += square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def constraint2():
        """
        Terminals must be included.
        """
        q = defaultdict(int)
        offset = 0
        for v in terminals:
            coef2 = [0] * qSize
            for i in range(0, maxDepth):
                coef2[get(var="xv", depth=i, param1=v)] += 1
            q = addQubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["q"], size=qSize)
            offset += square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def constraint3():
        """
        Non-terminals can appear at most once -> must not exceed yv[v]
        """
        q = defaultdict(int)
        offset = 0
        for v in range(0, n):
            coef2 = [0] * qSize
            if (v in terminals):
                continue
            for i in range(0, maxDepth):
                coef2[get(var="xv", depth=i, param1=v)] += 1
            coef2[get(var="yv", param1=v)] -= 1
            q = addQubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=A)["q"], size=qSize)
            offset += square(coef=coef2, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def constraint4():
        """
        Each vertex can have at most 1 edge connected to it
        """
        q = defaultdict(int)
        offset = 0
        for v in range(0, n):
            for i in range(1, maxDepth):
                coef2 = [0] * qSize
                for u in range(0, n):
                    if (inEdgeList(u, v)):
                        coef2[get(var="xe", depth=i, param1=u, param2=v)] += 1
                coef2[get(var="xv", depth=i, param1=v)] -= 1
                q = addQubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=A)["q"], size=qSize)
                offset += square(coef=coef2, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def constraint5():
        """
        Edges must be included in a given order
        """
        q = defaultdict(int)
        offset = 0
        for (u, v) in edgeList:
            for i in range(1, maxDepth):
                q[(get("xe", i, u, v), get("xe", i, u, v))] += 2 * A
                q[(get("xe", i, u, v), get("xv", i - 1, u))] -= 1 * A
                q[(get("xe", i, u, v), get("xv", i, v))] -= 1 * A
        return {
            "q": q,
            "offset": offset
        }

    def constraint6():
        """
        Number of times edge (u, v) is included in the tree must exceed ye[u][v]
        """
        q = defaultdict(int)
        offset = 0
        for (u, v) in udiEdgeList:
            coef2 = [0] * qSize
            for i in range(1, maxDepth):
                coef2[(get(var="xe", depth=i, param1=u, param2=v))] += 1
                coef2[(get(var="xe", depth=i, param1=v, param2=u))] += 1
            coef2[(get(var="ye", param1=u, param2=v))] -= 1
            q = addQubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=A)["q"], size=qSize)
            offset += square(coef=coef2, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def objective():
        q = defaultdict(int)
        offset = 0
        for (u, v) in edgeList:
            for i in range(1, maxDepth):
                q[(get("xe", i, u, v), get("xe", i, u, v))] += B * g[u][v]['weight']
        return {
            "q": q,
            "offset": offset
        }

    # Add constraints and objective function to the QUBO
    q = addQubo(q1=q, q2=constraint1()["q"], size=qSize)
    offset += constraint1()["offset"]
    q = addQubo(q1=q, q2=constraint2()["q"], size=qSize)
    offset += constraint2()["offset"]
    q = addQubo(q1=q, q2=constraint3()["q"], size=qSize)
    offset += constraint3()["offset"]
    q = addQubo(q1=q, q2=constraint4()["q"], size=qSize)
    offset += constraint4()["offset"]
    q = addQubo(q1=q, q2=constraint5()["q"], size=qSize)
    offset += constraint5()["offset"]
    q = addQubo(q1=q, q2=constraint6()["q"], size=qSize)
    offset += constraint6()["offset"]
    q = addQubo(q1=q, q2=objective()["q"], size=qSize)
    offset += objective()["offset"]

    # Solve QUBO with D-Wave
    chainStrength = uniform_torque_compensation(
        bqm=BinaryQuadraticModel.from_qubo(q), prefactor=chainStrengthPrefactor)
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(q,
                                   chain_strength=chainStrength,
                                   num_reads=numReads,
                                   label='Steiner Tree Soltuion',
                                   annealing_time=annealing_time)
    dwave.inspector.show(response)

    reportFile.write("## Result\n")
    reportTable(reportFile=reportFile, response=response)
    success = 0
    sample = response.record.sample[0]
    result = response.record.energy[0] + offset
    for i in range(0, len(response.record.sample)):
        if (response.record.energy[i] + offset < result):
            sample = response.record.sample[i]
            result = response.record.energy[i] + offset
            success = 1
        elif (response.record.energy[i] + offset == result):
            success += response.record.num_occurrences[i]

    print(len(sample))
    penalty1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=sample)
    penalty2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=sample)
    penalty3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=sample)
    penalty4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=sample)
    penalty5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=sample)
    penalty6 = calculate(q=constraint6()["q"], offset=constraint6()["offset"], x=sample)
    print("Penalty 1: {}".format(penalty1))
    print("Penalty 2: {}".format(penalty2))
    print("Penalty 3: {}".format(penalty3))
    print("Penalty 4: {}".format(penalty4))
    print("Penalty 5: {}".format(penalty5))
    print("Penalty 6: {}".format(penalty6))
    objectiveVal = calculate(q=objective()["q"], offset=objective()["offset"], x=sample)
    if (penalty1 != 0
            or penalty2 != 0
            or penalty3 != 0
            or penalty4 != 0
            or penalty5 != 0
            or penalty6 != 0):
        print("Cannot find a Steiner tree in the given graph. Energy is {}".format(result))
    else:
        print("Minimum Steiner tree found with total weight = {}".format(result))
    print("Included edges:")
    ans = []
    for (u, v) in udiEdgeList:
        if (sample[get("ye", param1=u, param2=v)] == 1):
            print("({}, {})".format(u, v))
            ans.append((u, v))
    if (result != 15):
        success = 0
    print("Success rate: {}/{}".format(success, numReads))

    return {
        "ans": ans,
        "success_rate": success,
    }

def fowler(g, terminals,
          numReads=1000,
          __lambda=1,
          chainStrengthPrefactor=0.3,
          annealing_time=200):
    """
    https://www.researchgate.net/profile/Alex-Fowler/publication/322634540_Improved_QUBO_Formulations_for_D-Wave_Quantum_Computing/links/5a65547caca272a1581f2809/Improved-QUBO-Formulations-for-D-Wave-Quantum-Computing.pdf
    """
    # First initialization
    reportFile = open("steinerTree_response.md", "w")
    n = len(g.nodes)
    m = len(g.edges)
    edgeList = list(g.edges)
    root = terminals[0]
    edgeListRootSepc = list(filter(lambda x: x[1] != root, list(g.edges)))
    m1 = len(edgeListRootSepc)
    print(edgeListRootSepc)

    # Hamiltonian parameters
    B = 1
    A = __lambda * B

    # Get the position of a variable in the QUBO
    def get(var="e", param1=0, param2=0):
        """
        e[u][v]: edge (u, v) in the DAG created by chosen edges
        x[u][v]: u appears before v in the DAG's topology order
        """
        if (var == "e"):
            return edgeListRootSepc.index((param1, param2))
        if (var == "x"):
            return m1 + (2 * n - param1 - 1) * param1 // 2 + (param2 - param1 - 1)

    # Generate QUBO
    q = defaultdict(int)
    qSize = m1 + n * (n - 1) // 2
    offset = 0

    def constraint1():
        """
        Vertices must be in the correct order: \
        x[u][v] = 0 && x[v][w] = 0 => x[u][w] = 0 \
        x[u][v] = 0 && x[v][w] = 1 => x[u][w] = 0 \
        x[u][v] = 0 && x[v][w] = 1 => x[u][w] = 1 \
        x[u][v] = 1 && x[v][w] = 0 => x[u][w] = 0 \
        x[u][v] = 1 && x[v][w] = 0 => x[u][w] = 1 \
        x[u][v] = 1 && x[v][w] = 1 => x[u][w] = 1 \
        """
        q = defaultdict(int)
        offset = 0
        for u in range(0, n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    q[(get("x", u, w), get("x", u, w))] += A
                    q[(get("x", u, v), get("x", v, w))] += A
                    q[(get("x", u, v), get("x", u, w))] -= A
                    q[(get("x", u, w), get("x", v, w))] -= A
        return {
            "q": q,
            "offset": offset
        }

    def constraint2():
        """
        If edge (u, v) exist in the DAG, its topology order must be correct:
        """
        q = defaultdict(int)
        offset = 0
        for (u, v) in edgeListRootSepc:
            if (u >= v): continue
            if (u == root): continue
            q[(get("e", u, v), get("e", u, v))] += A
            q[(get("e", u, v), get("x", u, v))] -= A
            q[(get("e", v, u), get("x", u, v))] += A
        return {
            "q": q,
            "offset": offset
        }

    def constraint3():
        """
        Non-root terminals must have exactly 1 incoming edge
        """
        q = defaultdict(int)
        offset = 0
        for v in terminals[1:]:
            coef2 = [0] * qSize
            for u in g.adj[v]:
                coef2[get("e", u, v)] += 1
            q = addQubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["q"], size=qSize)
            offset += square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def constraint4():
        """
        Non-terminals must have no more than 1 incoming edge
        """
        q = defaultdict(int)
        offset = 0
        for v in range(0, n):
            if (v in terminals):
                continue
            for u in g.adj[v]:
                for w in g.adj[v]:
                    if (u == w): continue
                    q[(get("e", u, v), get("e", w, v))] += A * n
        return {
            "q": q,
            "offset": offset
        }

    def constraint5():
        """
        Non-terminals cannot be DAG root
        """
        q = defaultdict(int)
        offset = 0
        for v in range(0, n):
            if (v in terminals):
                continue
            coef1 = [0] * qSize
            coef2 = [0] * qSize
            for u in g.adj[v]:
                coef1[get("e", u, v)] -= 1
            freeCoef1 = 1
            for u in g.adj[v]:
                if (u == root): continue
                coef2[get("e", v, u)] += 1
            q = addQubo(q1=q, q2=mul(coef1=coef1, freeCoef1=freeCoef1,
                                     coef2=coef2,
                                     size=qSize, __lambda=A)["q"], size=qSize)
            offset += mul(coef1=coef1, freeCoef1=freeCoef1,
                            coef2=coef2,
                            size=qSize, __lambda=A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def objective():
        """See paper"""
        q = defaultdict(int)
        offset = 0
        for (u, v) in edgeListRootSepc:
            q[(get("e", u, v), get("e", u, v))] += B * g[u][v]['weight']

        return {
            "q": q,
            "offset": offset
        }

    # Add constraints and objective function to the QUBO
    q = addQubo(q1=q, q2=constraint1()["q"], size=qSize)
    offset += constraint1()["offset"]
    q = addQubo(q1=q, q2=constraint2()["q"], size=qSize)
    offset += constraint2()["offset"]
    q = addQubo(q1=q, q2=constraint3()["q"], size=qSize)
    offset += constraint3()["offset"]
    q = addQubo(q1=q, q2=constraint4()["q"], size=qSize)
    offset += constraint4()["offset"]
    q = addQubo(q1=q, q2=constraint5()["q"], size=qSize)
    offset += constraint5()["offset"]
    q = addQubo(q1=q, q2=objective()["q"], size=qSize)
    offset += objective()["offset"]

    # x = [1, 0, 1, 0, 0, 0,
    #      1, 1, 0,
    #      1, 0,
    #      0]
    # x = [0, 1, 0, 0, 0, 1,
    #      0, 1, 1,
    #      1, 1,
    #      0]

    # x = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    #      1, 0, 0, 0, 1,
    #      0, 0, 0, 0,
    #      0, 0, 1,
    #      0, 1,
    #      1]

    # # print("Energy is {}".format(calculate(q, offset, x=x)))
    # print("Penalty 1 is {}".format(calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=x)))
    # print("Penalty 2 is {}".format(calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=x)))
    # print("Penalty 3 is {}".format(calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=x)))
    # print("Penalty 4 is {}".format(calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=x)))
    # print("Penalty 5 is {}".format(calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=x)))
    # print("Objective is {}".format(calculate(q=objective()["q"], offset=objective()["offset"], x=x)))

    # Solve QUBO with D-Wave
    chainStrength = uniform_torque_compensation(
        bqm=BinaryQuadraticModel.from_qubo(q), prefactor=chainStrengthPrefactor)
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(q,
                                   chain_strength=chainStrength,
                                   num_reads=numReads,
                                   label='Steiner Tree Soltuion',
                                   annealing_time=annealing_time)
    # dwave.inspector.show(response)

    reportFile.write("## Result\n")
    reportTable(reportFile=reportFile, response=response)
    success = 0
    sample = response.record.sample[0]
    result = response.record.energy[0] + offset
    for i in range(0, len(response.record.sample)):
        if (response.record.energy[i] + offset < result):
            sample = response.record.sample[i]
            result = response.record.energy[i] + offset
            success = 1
        elif (response.record.energy[i] + offset == result):
            success += response.record.num_occurrences[i]

    print(len(sample))
    penalty1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=sample)
    penalty2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=sample)
    penalty3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=sample)
    penalty4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=sample)
    penalty5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=sample)
    print("Penalty 1: {}".format(penalty1))
    print("Penalty 2: {}".format(penalty2))
    print("Penalty 3: {}".format(penalty3))
    print("Penalty 4: {}".format(penalty4))
    print("Penalty 5: {}".format(penalty5))
    objectiveVal = calculate(q=objective()["q"], offset=objective()["offset"], x=sample)
    if (penalty1 != 0
            or penalty2 != 0
            or penalty3 != 0
            or penalty4 != 0
            or penalty5 != 0):
        print("Cannot find a Steiner tree in the given graph. Energy is {}".format(result))
    else:
        print("Minimum Steiner tree found with total weight = {}".format(result))
    print("Included edges:")
    ans = []
    for (u, v) in edgeListRootSepc:
        if (sample[get("e", u, v)] == 1):
            print("({}, {})".format(u, v))
            ans.append((u, v))
    if (result != 15):
        success = 0
    print("Success rate: {}/{}".format(success, numReads))

    return {
        "ans": ans,
        "success_rate": success,
    }

def readInput(file):
    with open(file) as f:
        line = f.readline()
        n = int(line.split()[0])
        m = int(line.split()[1])
        g = nx.DiGraph()
        for i in range(0, m):
            line = f.readline()
            u = int(line.split()[0])
            v = int(line.split()[1])
            w = int(line.split()[2])
            g.add_edge(u, v, weight=w)
            g.add_edge(v, u, weight=w)
        line = f.readline()
        k = int(line)
        terminals = []
        line = f.readline()
        for i in range(0, k):
            terminals.append(int(line.split()[i]))
    return g, terminals

# g, terminals = readInput("steiner.inp")
# print(fowler(g=g, terminals=terminals,
#             numReads=1000,
#             __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1,
#             chainStrengthPrefactor=0.3,
#             annealing_time=200))