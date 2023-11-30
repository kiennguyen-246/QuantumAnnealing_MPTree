import networkx as nx

from src.successCounter import counter
from steinerTree import lucas, fowler


def makeGraph(v_e_list=None,
              seqList=None,
              terminals=None):
    copy = []
    for i in range(0, len(seqList)):
        if (seqList[i] not in copy):
            copy.append(seqList[i])
    seqList = copy
    if v_e_list is not None:
        g = nx.DiGraph()
        for i in v_e_list:
            [u, v, w] = i
            u1 = convertSeqToIndex(u, seqList)
            v1 = convertSeqToIndex(v, seqList)
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
    return {'graph': g1, 'terminals': terminals}

def getAns(v_e_list=None,
           seqList=None,
           terminals=None):
    g = makeGraph(v_e_list=v_e_list,
                               seqList=seqList,
                               terminals=terminals)["graph"]
    terminals = makeGraph(v_e_list=v_e_list,
                            seqList=seqList,
                            terminals=terminals)["terminals"]

    print("Graph: ", g)
    print("Terminals", terminals)

    print("\n\n\n\n\n------------------------\n")
    print("Success count: ")
    # counter(g=g, terminals=terminals, n_trials=10)
    print("\n\n\n\n\n------------------------\n")

    ans = fowler(g=g, terminals=terminals,
                numReads=1000,
                __lambda=len(g.nodes) * max([g[u][v]['weight'] for (u, v) in g.edges]) + 1,
                chainStrengthPrefactor=0.3,
                annealing_time=200)["ans"]
    print(ans)
    for i in range(0, len(ans)):
        ans[i] = (seqList[ans[i][0]], seqList[ans[i][1]])
    return ans
    


def convertSeqToIndex(seq, seqList):
    for i in range(0, len(seqList)):
        if (seq == seqList[i]):
            return i
