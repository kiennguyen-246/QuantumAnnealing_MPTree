import networkx as nx

from src.successCounter import counter
from steinerTree import *


def makeGraph(v_e_list=None,
              seqList=None,
              terminals=None):
    copy = []
    for i in range(0, len(seqList)):
        if (seqList[i] not in copy):
            copy.append(seqList[i])
    seqList = copy
    print("seqList = ", seqList)
    if v_e_list is not None:
        g = nx.DiGraph()
        for i in v_e_list:
            [u, v, w] = i
            u1 = convertSeqToIndex(u, seqList)
            v1 = convertSeqToIndex(v, seqList)
            if (u1 == v1):
                continue
            g.add_edge(u1, v1, weight=w)
            g.add_edge(v1, u1, weight=w)
        terminals = [convertSeqToIndex(i, seqList) for i in terminals]
    g1 = nx.DiGraph()
    for i in range(0, len(g.nodes)):
        g1.add_node(i)
    # print("g = ", g)
    for u in range(0, len(g.nodes)):
        for v in g.adj[u]:
            isDirect = True
            for w in range(0, len(g.nodes)):
                if (w != u and w != v and g[u][w]['weight'] + g[w][v]['weight'] == g[u][v]['weight']):
                    isDirect = False
                    break
            if (isDirect):
                g1.add_edge(u, v, weight=g[u][v]['weight'])
    # print("g1 = ", g1)
    root = terminals[0]

    msp = max([nx.shortest_path_length(g1, root, v, "weight") for v in terminals])
    for u in terminals:
        umsp = max([nx.shortest_path_length(g1, u, v, "weight") for v in terminals])
        if umsp < msp:
            msp = umsp
            root = u
    g2 = nx.DiGraph()
    for u in range(0, len(g1.nodes)):
        g2.add_node(u)
    for (u, v) in g1.edges:
        if (g[u][v]['weight'] <= msp):
            g2.add_edge(u, v, weight=g1[u][v]['weight'])

    print(len(g1.edges))
    print(len(g2.edges))
    for u in terminals:
        print(u, nx.shortest_path_length(g1, root, u, "weight"))
    return {'graph': g2, 'terminals': terminals, 'root': root}


def getAns(v_e_list=None,
           seqList=None,
           terminals=None):
    copy = []
    for i in range(0, len(seqList)):
        if (seqList[i] not in copy):
            copy.append(seqList[i])
    seqList = copy
    make_graph_res = makeGraph(v_e_list=v_e_list,
                               seqList=seqList,
                               terminals=terminals)
    (g, terminals, root) = (make_graph_res['graph'], make_graph_res['terminals'], make_graph_res['root'])

    print("Graph: ", g)
    print("Terminals", terminals)
    print("Root", root)

    print("\n\n\n\n\n------------------------\n")
    print("Success count: ")
    # counter(g=g, terminals=terminals, n_trials=10)
    print("\n\n\n\n\n------------------------\n")

    ans = fowler(g=g, terminals=terminals, root=root,
                 numReads=1000,
                 __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1,
                 chainStrengthPrefactor=0.3,
                 annealing_time=200)["ans"]

    # ans = sridhar_lam_blelloch_ravi_schwartz_ilp(g=g, terminals=terminals, root=root)["ans"]

    print(ans)
    ans_edges = []
    for i in range(0, len(ans)):
        ans_edges.append((seqList[ans[i][0]], seqList[ans[i][1]], ans[i][2]))

    # def hamming_distance(u, v):
    #     return sum(c1 != c2 for c1, c2 in zip(u, v))
    # for i in range(0, len(ans)):
    #     ans_edges.append((seqList[ans[i][0]], seqList[ans[i][1]], hamming_distance(seqList[ans[i][0]], seqList[ans[i][1]])))




    return ans_edges


# def newport(edges):
#     g = nx.Graph()
#     for i in range(0, len(edges)):
#         g.add_edge(edges[i][0], edges[i][1], weight=edges[i][2])
#     post_order = []
#     dist = [-1] * max(g.nodes)
#     root = min(g.nodes)
#     dist[root] = 0
#
#     def dfs(u):
#         dist[u] = 0
#         for v in g.adj[u]:
#             if dist[v] != -1:
#                 continue
#             dfs(v)
#         post_order.append(u)
#
#     dfs(root)
#     print(post_order)


def convertSeqToIndex(seq, seqList):
    for i in range(0, len(seqList)):
        if (seq == seqList[i]):
            return i


# newport([(0, 5, 1), (0, 10, 1), (5, 4, 3), (8, 2, 1), (8, 3, 2), (10, 1, 2), (10, 8, 1)])