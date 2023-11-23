import networkx as nx
from steinerTree import lucas


def makeGraph(v_e_list=None,
              seqList=None,
              terminals=None):
    if v_e_list is not None:
        g = nx.DiGraph()
        for i in v_e_list:
            [u, v, w] = i
            u1 = convertSeqToIndex(u, seqList)
            v1 = convertSeqToIndex(v, seqList)
            g.add_edge(u1, v1, weight=w)
            g.add_edge(v1, u1, weight=w)
        terminals = [convertSeqToIndex(i, seqList) for i in terminals]
    return {'graph': g, 'terminals': terminals}


def getAns(v_e_list=None,
           seqList=None,
           terminals=None):
    g = makeGraph(v_e_list=v_e_list,
                               seqList=seqList,
                               terminals=terminals)["graph"]
    terminals = makeGraph(v_e_list=v_e_list,
                            seqList=seqList,
                            terminals=terminals)["terminals"]

    print(g)
    print(terminals)
    ans = lucas(g=g, terminals=terminals,
                numReads=1000,
                __lambda=1,
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
