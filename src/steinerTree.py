import json
import os
import time

import dimod
from dwave.system import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import LazyFixedEmbeddingComposite
from dwave.embedding.chain_strength import uniform_torque_compensation
from dimod.binary import BinaryQuadraticModel
from minorminer.utils import DisconnectedChainError
from neal import SimulatedAnnealingSampler
import dwave.inspector

from solve_bqm import *
from utils import *
import numpy as np
from gekko import GEKKO

output_path = "output/"


def lucas(g, terminals,
          num_reads=1000,
          __lambda=1):
    """
    https://www.frontiersin.org/articles/10.3389/fphy.2014.00005/full
    """
    # First initialization
    reportFile = open("steinerTree_response.md", "w")
    n = len(g.nodes)
    m = len(g.edges)
    edgeList = list(g.edges)
    udiEdgeList = list(filter(lambda x: x[0] < x[1], edgeList))
    method = "L_"

    def inEdgeList(u, v):
        return (u, v) in edgeList

    maxDepth = len(terminals) // 2 + 1

    # Hamiltonian parameters
    B = 1
    A = __lambda * B

    var_map = []
    for v in range(0, n):
        for i in range(0, maxDepth):
            var_map.append(("xv", i, v))
        if  v not in terminals:
            var_map.append(("yv", v))
    for (u, v) in edgeList:
        for i in range(1, maxDepth):
            var_map.append(("xe", i, u, v))
        if u < v:
            var_map.append(("ye", u, v))


    # Get the position of a variable in the QUBO
    def get(var="xv", depth=0, param1=0, param2=0):
        """ xv[i][x]: vertex x at depth i \\
            xe[i][x][y]: edge (x, y), y at depth i and x at depth i - 1 \\
            yv[x]: vertex x is in the tree \\
            ye[x][y]: edge (x, y) is in the tree"""
        # if (var == "xv"):
        #     if (depth == 0):
        #         return param1
        #     return n + (depth - 1) * (n + m) + param1
        # if (var == "xe"):
        #     if (inEdgeList(param1, param2) == False):
        #         return -1
        #     return n + (depth - 1) * (n + m) + n + edgeList.index((param1, param2))
        # if (var == "yv"):
        #     return n + (maxDepth - 1) * (n + m) + param1
        # if (var == "ye"):
        #     if (inEdgeList(param1, param2) == False):
        #         return -1
        #     return n + (maxDepth - 1) * (n + m) + n + udiEdgeList.index((param1, param2))
        if var == "xv":
            return var_map.index((var, depth, param1))
        if var == "xe":
            return var_map.index((var, depth, param1, param2))
        if var == "yv":
            return var_map.index((var, param1))
        if var == "ye":
            return var_map.index((var, param1, param2))


    # Generate QUBO
    q = defaultdict(int)
    qSize = len(var_map)
    print(len(var_map))
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
        q = add_qubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["q"], size=qSize)
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
            q = add_qubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=A)["q"], size=qSize)
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
            q = add_qubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=5 * A)["q"], size=qSize)
            offset += square(coef=coef2, size=qSize, __lambda=5 * A)["offset"]
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
                q = add_qubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=A)["q"], size=qSize)
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
            q = add_qubo(q1=q, q2=square(coef=coef2, size=qSize, __lambda=A)["q"], size=qSize)
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
    q = add_qubo(q1=q, q2=constraint1()["q"], size=qSize)
    offset += constraint1()["offset"]
    q = add_qubo(q1=q, q2=constraint2()["q"], size=qSize)
    offset += constraint2()["offset"]
    q = add_qubo(q1=q, q2=constraint3()["q"], size=qSize)
    offset += constraint3()["offset"]
    q = add_qubo(q1=q, q2=constraint4()["q"], size=qSize)
    offset += constraint4()["offset"]
    q = add_qubo(q1=q, q2=constraint5()["q"], size=qSize)
    offset += constraint5()["offset"]
    q = add_qubo(q1=q, q2=constraint6()["q"], size=qSize)
    offset += constraint6()["offset"]
    q = add_qubo(q1=q, q2=objective()["q"], size=qSize)
    offset += objective()["offset"]

    print(q)
    for i in range(0, len(var_map)):
        in_q = False
        for j in range(0, len(var_map)):
            if (i, j) in q:
                in_q = True
                break
        if (in_q == False):
            print(i, var_map[i])

    # Solve QUBO with D-Wave
    bqm = BinaryQuadraticModel.from_qubo(q, offset)
    print("Number of non-zero elements in QUBO matrix: {}".format(len(q)))
    while True:
        try:
            # Solve QUBO with D-Wave
            response = solve_quantum_annealing(bqm=bqm, method=method, num_reads=num_reads)

            # # Solve QUBO with Simulated Annealing
            # response = solve_simulated_annealing(bqm=bqm, method=method, num_reads=num_reads)
        except RuntimeError as e:
            print("Exception caught", e)
            continue
        break

    reportFile.write("## Result\n")
    reportTable(reportFile=reportFile, response=response)
    optimal = 0
    success = 0
    sample = response.record.sample[0]
    result = 1e9
    for i in range(0, len(response.record.sample)):
        if response.record.energy[i] < result:
            sample = response.record.sample[i]
            result = response.record.energy[i]
            optimal = response.record.num_occurrences[i]
        elif response.record.energy[i] == result:
            optimal += response.record.num_occurrences[i]
        pen1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=response.record.sample[i])
        pen2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=response.record.sample[i])
        pen3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=response.record.sample[i])
        pen4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=response.record.sample[i])
        pen5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=response.record.sample[i])
        pen6 = calculate(q=constraint6()["q"], offset=constraint6()["offset"], x=response.record.sample[i])
        if (pen1 == 0
                and pen2 == 0
                and pen3 == 0
                and pen4 == 0
                and pen5 == 0
                and pen6 == 0):
            success += response.record.num_occurrences[i]
            # print(pen1, pen2, pen3, pen4, pen5, pen6)
    optimal_val = ilp(g, terminals, terminals[0])["objective"]
    # print("Comparison:", result, optimal_val)
    if result != optimal_val:
        optimal = 0

    penalty1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=sample)
    penalty2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=sample)
    penalty3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=sample)
    penalty4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=sample)
    penalty5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=sample)
    penalty6 = calculate(q=constraint6()["q"], offset=constraint6()["offset"], x=sample)
    objectiveVal = calculate(q=objective()["q"], offset=objective()["offset"], x=sample)
    print("Penalty 1: {}".format(penalty1))
    print("Penalty 2: {}".format(penalty2))
    print("Penalty 3: {}".format(penalty3))
    print("Penalty 4: {}".format(penalty4))
    print("Penalty 5: {}".format(penalty5))
    print("Penalty 6: {}".format(penalty6))
    print("Objective: {}".format(objectiveVal))
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
            ans.append((u, v, g[u][v]['weight']))
    print("Steiner tree creation rate: {}/{}".format(success, num_reads))
    print("Optimal rate: {}/{}".format(optimal, num_reads))

    print(ans)

    ans_dict = {
        "ans": ans,
        "energy": result,
        "non_zero": len(q),
        "success_rate": int(success),
        "optimal_rate": int(optimal),
    }

    data_name = os.environ.get("PHYLO_FILE")
    output_dir = "output/" + data_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + os.getenv("SOLVER_CONFIG") + "_2.json", "w") as f:
        json.dump(ans_dict, f, indent=4)

    return ans_dict


def fowler(g, terminals, root=0,
           __lambda=1):
    """
    https://www.researchgate.net/profile/Alex-Fowler/publication/322634540_Improved_QUBO_Formulations_for_D-Wave_Quantum_Computing/links/5a65547caca272a1581f2809/Improved-QUBO-Formulations-for-D-Wave-Quantum-Computing.pdf
    """
    num_reads = 1000
    method = "F_"

    # First initialization
    report_file = open("steinerTree_response.md", "w")
    n = len(g.nodes)
    m = len(g.edges)
    edge_list = list(g.edges)
    root = terminals[0]
    edge_list_root_sepc = list(filter(lambda x: x[1] != root, list(g.edges)))
    m1 = len(edge_list_root_sepc)
    # print(edge_list)
    print(edge_list_root_sepc)

    # Hamiltonian parameters
    B = 1
    A = __lambda * B

    var_map = []
    for (u, v) in edge_list_root_sepc:
        var_map.append(("e", u, v))
    for u in range(0, n):
        for v in range(u + 1, n):
            if u != root and v != root:
                var_map.append(("x", u, v))
    # for v in g.nodes:
    #     if v in terminals and v == root:
    #         continue
    #     for i in range(1, len(g.adj[v])):
    #         var_map.append(("s", v, i))
    #         if v not in terminals and i != len(g.adj[v]) - 1:
    #             var_map.append(("t", v, i))

    # Get the position of a variable in the QUBO
    def get(var="e", param1=0, param2=0):
        """
        e[u][v]: edge (u, v) in the DAG created by chosen edges
        x[u][v]: u appears before v in the DAG's topology order
        s[v][i]: sum of e[u][v] to the i-th adjacent vertex of v
        """
        return var_map.index((var, param1, param2))

    # Generate QUBO
    q = defaultdict(int)
    qSize = len(var_map)
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
        ext_coef = 1
        for u in range(0, n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    if (u == root or v == root or w == root): continue
                    q[(get("x", u, w), get("x", u, w))] += ext_coef * A
                    q[(get("x", u, v), get("x", v, w))] += ext_coef * A
                    q[(get("x", u, v), get("x", u, w))] -= ext_coef * A
                    q[(get("x", u, w), get("x", v, w))] -= ext_coef * A
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
        ext_coef = 1
        for (u, v) in edge_list_root_sepc:
            if (u >= v): continue
            if (u == root or v == root): continue
            q[(get("e", u, v), get("e", u, v))] += ext_coef * A
            q[(get("e", u, v), get("x", u, v))] -= ext_coef * A
            q[(get("e", v, u), get("x", u, v))] += ext_coef * A
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
        ext_coef = 1
        for v in terminals:
            if (v == root): continue
            coef2 = [0] * qSize
            for u in g.adj[v]:
                coef2[get("e", u, v)] += 1
            q = add_qubo(q1=q, q2=square(coef=coef2, freeCoef=-1, size=qSize, __lambda=ext_coef * A)["q"],
                         size=qSize)
            offset += square(coef=coef2, freeCoef=-1, size=qSize, __lambda=ext_coef * A)["offset"]
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
        ext_coef = 1
        for v in range(0, n):
            if v in terminals:
                continue
            for u in g.adj[v]:
                for w in g.adj[v]:
                    if u == w:
                        continue
                    q[(get("e", u, v), get("e", w, v))] += ext_coef * A * n
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
        ext_coef = 1
        for v in range(0, n):
            if v in terminals:
                continue
            coef1 = [0] * qSize
            coef2 = [0] * qSize
            for u in g.adj[v]:
                coef1[get("e", u, v)] -= 1
            freeCoef1 = 1
            for u in g.adj[v]:
                if (u == root): continue
                coef2[get("e", v, u)] += 1
            q = add_qubo(q1=q, q2=mul(coef1=coef1, freeCoef1=freeCoef1,
                                      coef2=coef2,
                                      size=qSize, __lambda=ext_coef * A)["q"], size=qSize)
            offset += mul(coef1=coef1, freeCoef1=freeCoef1,
                          coef2=coef2,
                          size=qSize, __lambda=ext_coef * A)["offset"]
        return {
            "q": q,
            "offset": offset
        }

    def objective():
        """See paper"""
        q = defaultdict(int)
        offset = 0
        for (u, v) in edge_list_root_sepc:
            q[(get("e", u, v), get("e", u, v))] += B * g[u][v]['weight']

        return {
            "q": q,
            "offset": offset
        }

    # Add constraints and objective function to the QUBO
    q = add_qubo(q1=q, q2=constraint1()["q"], size=qSize)
    offset += constraint1()["offset"]
    q = add_qubo(q1=q, q2=constraint2()["q"], size=qSize)
    offset += constraint2()["offset"]
    q = add_qubo(q1=q, q2=constraint3()["q"], size=qSize)
    offset += constraint3()["offset"]
    q = add_qubo(q1=q, q2=constraint4()["q"], size=qSize)
    offset += constraint4()["offset"]
    q = add_qubo(q1=q, q2=constraint5()["q"], size=qSize)
    offset += constraint5()["offset"]
    q = add_qubo(q1=q, q2=objective()["q"], size=qSize)
    offset += objective()["offset"]

    print("Number of non-zero elements in QUBO matrix: {}".format(len(q)))
    bqm = BinaryQuadraticModel.from_qubo(q, offset)
    # print(q)
    # print(bqm)

    fixed_var_map = var_map.copy()
    # print(var_map)
    # print(fixed_var_map)

    while True:
        try:
            # Solve QUBO with D-Wave
            response = solve_quantum_annealing(bqm=bqm, method=method, num_reads=num_reads)

            # # Solve QUBO with Simulated Annealing
            # response = solve_simulated_annealing(bqm=bqm, method=method, num_reads=num_reads)
        except RuntimeError as e:
            print("Exception caught", e)
            continue
        break

    # Analyze result
    report_file.write("## Result\n")
    reportTable(reportFile=report_file, response=response)
    success = 0
    optimal = 0
    optimal_val = ilp(g, terminals, root)["objective"]
    sample = response.record.sample[0]
    result = 1e9
    satisfy1 = 0
    satisfy2 = 0
    satisfy3 = 0
    satisfy4 = 0
    satisfy5 = 0
    for i in range(0, len(response.record.sample)):
        cur_sample = response.record.sample[i]
        x = []
        for j in range(0, len(var_map)):
            if var_map[j] in fixed_var_map:
                x.append(cur_sample[fixed_var_map.index(var_map[j])])
            else:
                x.append(1)
        # print(x)
        pen1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=x)
        pen2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=x)
        pen3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=x)
        pen4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=x)
        pen5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=x)
        obj = calculate(q=objective()["q"], offset=objective()["offset"], x=x)
        # energy = pen1 + pen2 + pen3 + pen4 + pen5 + obj
        if (pen1 == 0
                and pen2 == 0
                and pen3 == 0
                and pen4 == 0
                and pen5 == 0):
            success += response.record.num_occurrences[i]
        if pen1 == 0:
            satisfy1 += response.record.num_occurrences[i]
        if pen2 == 0:
            satisfy2 += response.record.num_occurrences[i]
        if pen3 == 0:
            satisfy3 += response.record.num_occurrences[i]
        if pen4 == 0:
            satisfy4 += response.record.num_occurrences[i]
        if pen5 == 0:
            satisfy5 += response.record.num_occurrences[i]
        # print(pen1, pen2, pen3, pen4, pen5, obj, response.record.energy[i])
        if response.record.energy[i] < result:
            sample = response.record.sample[i]
            result = response.record.energy[i]
            optimal = response.record.num_occurrences[i]
        elif response.record.energy[i] == result:
            optimal += response.record.num_occurrences[i]
        if result != optimal_val:
            optimal = 0
    print(result)

    # print(len(sample), len(var_map))
    # sample = [1, 0, 1, 0, 0, 0, 1, 1, 1, 1]
    # sample = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    # print(g.adj[3])
    # print(g.adj[5])
    # sample = [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]

    x = []
    for j in range(0, len(var_map)):
        if var_map[j] in fixed_var_map:
            x.append(sample[fixed_var_map.index(var_map[j])])
        else:
            x.append(1)
    # x[get("e", 0, 1)] = 1

    # for i in range(0, len(sample)):
    #     print(fixed_var_map[i], "=", sample[i])
    penalty1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=x)
    penalty2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=x)
    penalty3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=x)
    penalty4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=x)
    penalty5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=x)
    objectiveVal = calculate(q=objective()["q"], offset=objective()["offset"], x=x)
    print("Penalty 1: {}".format(penalty1))
    print("Penalty 2: {}".format(penalty2))
    print("Penalty 3: {}".format(penalty3))
    print("Penalty 4: {}".format(penalty4))
    print("Penalty 5: {}".format(penalty5))
    print("Objective: {}".format(objectiveVal))
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
    for (u, v) in edge_list_root_sepc:
        if (sample[get("e", u, v)] == 1):
            print("({}, {})".format(u, v))
            ans.append((u, v, g[u][v]['weight']))
    # if (result != 13):
    #     optimal = 0
    print("Steiner tree creation rate: {}/{}".format(success, num_reads))
    print("Satisfaction statistics:")
    print("- Constraint 1: {}/{}".format(satisfy1, num_reads))
    print("- Constraint 2: {}/{}".format(satisfy2, num_reads))
    print("- Constraint 3: {}/{}".format(satisfy3, num_reads))
    print("- Constraint 4: {}/{}".format(satisfy4, num_reads))
    print("- Constraint 5: {}/{}".format(satisfy5, num_reads))
    print("Optimal rate: {}/{}".format(optimal, num_reads))
    ans_dict = {
        "ans": ans,
        "energy": result,
        "non_zero": len(q),
        "success_rate": int(success),
        "optimal_rate": int(optimal),
        "satisfy_stats": [int(satisfy1), int(satisfy2), int(satisfy3), int(satisfy4), int(satisfy5)],
    }
    data_name = os.environ.get("PHYLO_FILE")
    output_dir = "output/" + data_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + os.getenv("SOLVER_CONFIG") + "_2.json", "w") as f:
        json.dump(ans_dict, f, indent=4)
    return ans_dict


def nghiem(g, terminals, root=0,
           __lambda=1):
    """
    Proposed formulation
    """
    num_reads = 1000
    method = "N_"

    # First initialization
    report_file = open("steinerTree_response.md", "w")
    n = len(g.nodes)
    m = len(g.edges)
    edge_list = list(g.edges)
    root = terminals[0]
    edge_list_root_sepc = list(filter(lambda x: x[1] != root, list(g.edges)))
    m1 = len(edge_list_root_sepc)
    # print(edge_list)
    print(edge_list_root_sepc)

    # Hamiltonian parameters
    B = 1
    A = __lambda * B

    var_map = []
    for (u, v) in edge_list_root_sepc:
        var_map.append(("e", u, v))
    for u in range(0, n):
        for v in range(u + 1, n):
            if u != root and v != root:
                var_map.append(("x", u, v))
    for v in g.nodes:
        if v in terminals and v == root:
            continue
        adj = list(g.adj[v])
        for i in range(1, len(adj)):
            var_map.append(("s", v, i))
        if root in adj:
            adj.remove(root)
        for i in range(1, len(adj)):
            if v not in terminals and i != len(adj) - 1:
                var_map.append(("t", v, i))

    # Get the position of a variable in the QUBO
    def get(var="e", param1=0, param2=0):
        """
        e[u][v]: edge (u, v) in the DAG created by chosen edges
        x[u][v]: u appears before v in the DAG's topology order
        s[v][i]: sum of e[u][v] to the i-th adjacent vertex of v
        """
        return var_map.index((var, param1, param2))

    # Generate QUBO
    q = defaultdict(int)
    qSize = len(var_map)
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
        ext_coef = 1
        for u in range(0, n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    if (u == root or v == root or w == root): continue
                    q[(get("x", u, w), get("x", u, w))] += ext_coef * A
                    q[(get("x", u, v), get("x", v, w))] += ext_coef * A
                    q[(get("x", u, v), get("x", u, w))] -= ext_coef * A
                    q[(get("x", u, w), get("x", v, w))] -= ext_coef * A
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
        ext_coef = 1
        for (u, v) in edge_list_root_sepc:
            if (u >= v): continue
            if (u == root or v == root): continue
            q[(get("e", u, v), get("e", u, v))] += ext_coef * A
            q[(get("e", u, v), get("x", u, v))] -= ext_coef * A
            q[(get("e", v, u), get("x", u, v))] += ext_coef * A
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
        ext_coef = 1

        for v in terminals:
            if v == root:
                continue
            adj = list(g.adj[v])
            if len(adj) <= 1:
                continue
            q = add_bitwise_or_exc_11(q=q, x=get("e", adj[0], v), y=get("e", adj[1], v), z=get("s", v, 1),
                                      __lambda=ext_coef * A)
            # print("s", v, 1)
            for i in range(2, len(adj)):
                q = add_bitwise_or_exc_11(q=q, x=get("s", v, i - 1), y=get("e", adj[i], v), z=get("s", v, i),
                                          __lambda=ext_coef * A)
                # print("s", v, i)
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
        ext_coef = 1

        for v in g.nodes:
            if v in terminals or v == root:
                continue
            adj = list(g.adj[v])
            if len(adj) == 1:
                continue
            q = add_bitwise_or_exc_11(q=q, x=get("e", adj[0], v), y=get("e", adj[1], v), z=get("s", v, 1),
                                      __lambda=ext_coef * A)
            # print("s", v, 1)
            for i in range(2, len(adj)):
                q = add_bitwise_or_exc_11(q=q, x=get("s", v, i - 1), y=get("e", adj[i], v), z=get("s", v, i),
                                          __lambda=ext_coef * A)
                # print("s", v, i)
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
        ext_coef = 1
        for v in g.nodes:
            if v in terminals or v == root:
                continue
            adj = list(g.adj[v])
            if root in adj:
                adj.remove(root)
            if len(adj) > 2:
                q = add_bitwise_or(q=q, x=get("e", v, adj[0]), y=get("e", v, adj[1]), z=get("t", v, 1),
                                   __lambda=ext_coef * A)
                # print("t", v, 1)
            elif len(adj) > 1:
                q = add_bitwise_or(q=q, x=get("e", v, adj[0]), y=get("e", v, adj[1]),
                                   z=get("s", v, len(g.adj[v]) - 1),
                                   __lambda=ext_coef * A)
                # print("s", v, len(g.adj[v]) - 1)
            else:
                q = add_bitwise_or(q=q, x=get("e", v, adj[0]), y=get("e", v, adj[0]),
                                   z=get("s", v, len(g.adj[v]) - 1),
                                   __lambda=ext_coef * A)
                # print("s", v, len(g.adj[v]) - 1)
            for i in range(2, len(adj)):
                if i != len(adj) - 1:
                    q = add_bitwise_or(q=q, x=get("t", v, i - 1), y=get("e", v, adj[i]), z=get("t", v, i),
                                       __lambda=ext_coef * A)
                    # print("t", v, i)
                else:
                    q = add_bitwise_or(q=q, x=get("t", v, i - 1), y=get("e", v, adj[i]),
                                       z=get("s", v, len(g.adj[v]) - 1),
                                       __lambda=ext_coef * A)
                    # print("s", v, len(g.adj[v]) - 1)
        return {
            "q": q,
            "offset": offset
        }

    def objective():
        """See paper"""
        q = defaultdict(int)
        offset = 0
        for (u, v) in edge_list_root_sepc:
            q[(get("e", u, v), get("e", u, v))] += B * g[u][v]['weight']

        return {
            "q": q,
            "offset": offset
        }

    # Add constraints and objective function to the QUBO
    q = add_qubo(q1=q, q2=constraint1()["q"], size=qSize)
    offset += constraint1()["offset"]
    q = add_qubo(q1=q, q2=constraint2()["q"], size=qSize)
    offset += constraint2()["offset"]
    q = add_qubo(q1=q, q2=constraint3()["q"], size=qSize)
    offset += constraint3()["offset"]
    q = add_qubo(q1=q, q2=constraint4()["q"], size=qSize)
    offset += constraint4()["offset"]
    q = add_qubo(q1=q, q2=constraint5()["q"], size=qSize)
    offset += constraint5()["offset"]
    q = add_qubo(q1=q, q2=objective()["q"], size=qSize)
    offset += objective()["offset"]

    print("Number of non-zero elements in QUBO matrix: {}".format(len(q)))
    bqm = BinaryQuadraticModel.from_qubo(q, offset)
    # print(q)
    # print(bqm)

    fixed_var_map = var_map.copy()
    for v in terminals:
        if v != root:
            adj = list(g.adj[v])
            print(adj)
            if len(g.adj[v]) == 1:
                # print("e", g.adj[v][0], v)
                print(adj[0])
                bqm.fix_variable(v=get("e", adj[0], v), value=1)
                fixed_var_map.remove(("e", adj[0], v))
            else:
                # print("s", v, len(g.adj[v]) - 1)
                bqm.fix_variable(v=get("s", v, len(adj) - 1), value=1)
                fixed_var_map.remove(("s", v, len(adj) - 1))
    # print(len(var_map))
    # print(var_map)
    # print(len(fixed_var_map))
    # print(fixed_var_map)

    while True:
        try:
            # Solve QUBO with D-Wave
            response = solve_quantum_annealing(bqm=bqm, method=method, num_reads=num_reads)

            # # Solve QUBO with Simulated Annealing
            # response = solve_simulated_annealing(bqm=bqm, method=method, num_reads=num_reads)
        except RuntimeError as e:
            print("Exception caught", e)
            continue
        break

    # Analyze result
    report_file.write("## Result\n")
    reportTable(reportFile=report_file, response=response)
    success = 0
    optimal = 0
    optimal_val = ilp(g, terminals, root)["objective"]
    sample = response.record.sample[0]
    result = 1e9
    satisfy1 = 0
    satisfy2 = 0
    satisfy3 = 0
    satisfy4 = 0
    satisfy5 = 0
    for i in range(0, len(response.record.sample)):
        cur_sample = response.record.sample[i]
        x = []
        for j in range(0, len(var_map)):
            if var_map[j] in fixed_var_map:
                x.append(cur_sample[fixed_var_map.index(var_map[j])])
            else:
                x.append(1)
        # print(x)
        pen1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=x)
        pen2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=x)
        pen3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=x)
        pen4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=x)
        pen5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=x)
        obj = calculate(q=objective()["q"], offset=objective()["offset"], x=x)
        # energy = pen1 + pen2 + pen3 + pen4 + pen5 + obj
        if (pen1 == 0
                and pen2 == 0
                and pen3 == 0
                and pen4 == 0
                and pen5 == 0):
            success += response.record.num_occurrences[i]
        if pen1 == 0:
            satisfy1 += response.record.num_occurrences[i]
        if pen2 == 0:
            satisfy2 += response.record.num_occurrences[i]
        if pen3 == 0:
            satisfy3 += response.record.num_occurrences[i]
        if pen4 == 0:
            satisfy4 += response.record.num_occurrences[i]
        if pen5 == 0:
            satisfy5 += response.record.num_occurrences[i]
        # print(pen1, pen2, pen3, pen4, pen5, obj, response.record.energy[i])
        if response.record.energy[i] < result:
            sample = response.record.sample[i]
            result = response.record.energy[i]
            optimal = response.record.num_occurrences[i]
        elif response.record.energy[i] == result:
            optimal += response.record.num_occurrences[i]
        if result != optimal_val:
            optimal = 0
    print(result)

    # print(len(sample), len(var_map))
    # sample = [1, 0, 1, 0, 0, 0, 1, 1, 1, 1]
    # sample = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    # print(g.adj[3])
    # print(g.adj[5])
    # sample = [1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]

    x = []
    for j in range(0, len(var_map)):
        if var_map[j] in fixed_var_map:
            x.append(sample[fixed_var_map.index(var_map[j])])
        else:
            x.append(1)
    # x[get("e", 0, 1)] = 1

    # for i in range(0, len(sample)):
    #     print(fixed_var_map[i], "=", sample[i])
    penalty1 = calculate(q=constraint1()["q"], offset=constraint1()["offset"], x=x)
    penalty2 = calculate(q=constraint2()["q"], offset=constraint2()["offset"], x=x)
    penalty3 = calculate(q=constraint3()["q"], offset=constraint3()["offset"], x=x)
    penalty4 = calculate(q=constraint4()["q"], offset=constraint4()["offset"], x=x)
    penalty5 = calculate(q=constraint5()["q"], offset=constraint5()["offset"], x=x)
    objectiveVal = calculate(q=objective()["q"], offset=objective()["offset"], x=x)
    print("Penalty 1: {}".format(penalty1))
    print("Penalty 2: {}".format(penalty2))
    print("Penalty 3: {}".format(penalty3))
    print("Penalty 4: {}".format(penalty4))
    print("Penalty 5: {}".format(penalty5))
    print("Objective: {}".format(objectiveVal))
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
    for (u, v) in edge_list_root_sepc:
        if (sample[get("e", u, v)] == 1):
            print("({}, {})".format(u, v))
            ans.append((u, v, g[u][v]['weight']))
    # if (result != 13):
    #     optimal = 0
    print("Steiner tree creation rate: {}/{}".format(success, num_reads))
    print("Satisfaction statistics:")
    print("- Constraint 1: {}/{}".format(satisfy1, num_reads))
    print("- Constraint 2: {}/{}".format(satisfy2, num_reads))
    print("- Constraint 3: {}/{}".format(satisfy3, num_reads))
    print("- Constraint 4: {}/{}".format(satisfy4, num_reads))
    print("- Constraint 5: {}/{}".format(satisfy5, num_reads))
    print("Optimal rate: {}/{}".format(optimal, num_reads))
    ans_dict = {
        "ans": ans,
        "energy": result,
        "non_zero": len(q),
        "success_rate": int(success),
        "optimal_rate": int(optimal),
        "satisfy_stats": [int(satisfy1), int(satisfy2), int(satisfy3), int(satisfy4), int(satisfy5)],
    }
    data_name = os.environ.get("PHYLO_FILE")
    output_dir = "output/" + data_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + os.getenv("SOLVER_CONFIG") + "_2.json", "w") as f:
        json.dump(ans_dict, f, indent=4)

    return ans_dict


def ilp(g, terminals, root=0):
    print("Running ILP")
    print(g)
    print(g.edges)
    print(terminals)
    print("Root: ", root)

    model = GEKKO(remote=False)
    model.options.SOLVER = 1
    n = len(g.nodes)
    m = len(g.edges)
    edgelist = list(g.edges)
    variables = []
    for (u, v) in g.edges:
        variables.append({
            "label": ("s", u, v),
            "var": model.Var(lb=0, ub=1, integer=True)
        })
    for t in terminals:
        for (u, v) in g.edges:
            variables.append({
                "label": ("f", t, u, v),
                "var": model.Var(lb=0, ub=1)
            })

    def get(type, param1=0, param2=0, param3=0):
        if type == "s":
            for var in variables:
                if var["label"] == (type, param1, param2):
                    return var["var"]
            return None
        if type == "f":
            for var in variables:
                if var["label"] == (type, param1, param2, param3):
                    return var["var"]
            return None

    # print([(("f", 0, 1, v), get("f", 0, 1, v)) for v in g.adj[1]])
    # print([(("f", 0, v, 1), get("f", 0, v, 1)) for v in g.adj[1]])

    # Constraint 1
    for t in terminals:
        for u in g.nodes:
            if u != t and u != root:
                model.Equation(np.sum([get("f", t, u, v) for v in g.adj[u]])
                               == np.sum([get("f", t, v, u) for v in g.adj[u]]))

    # Constraint 2
    for t in terminals:
        if t == root:
            continue
        model.Equation(np.sum([get("f", t, v, t) for v in g.adj[t]]) == 1)
        model.Equation(np.sum([get("f", t, t, v) for v in g.adj[t]]) == 0)
        model.Equation(np.sum([get("f", t, root, v) for v in g.adj[root]]) == 1)

    # Constraint 3
    for t in terminals:
        for (u, v) in g.edges:
            model.Equation(get("f", t, u, v) <= get("s", u, v))

    def objective_function():
        ret = 0
        for (u, v) in edgelist:
            ret += g[u][v]['weight'] * get("s", u, v)
        return ret

    model.Obj(objective_function())
    model.solve(disp=False)
    print(f"Objective: {model.options.objfcnval}")
    print("Included edges:")
    ans = []
    for (u, v) in g.edges:
        if get("s", u, v).value[0] == 1:
            print("({}, {})".format(u, v))
            ans.append((u, v, g[u][v]['weight']))
    # print("Variables: ")
    # for var in variables:
    #     print(var["label"], var["var"].value)
    print("----------------------\n\n")
    return {
        "ans": ans,
        "objective": model.options.objfcnval
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


g, terminals = readInput("steiner.inp")
# lucas(g=g, terminals=terminals,
#        __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1)
# nghiem(g=g, terminals=terminals, root=terminals[0],
#        __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1)
