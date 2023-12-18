from collections import Counter
import os
from itertools import combinations
import networkx as nx
from makeGraph import getAns

def read_input(file_path):
    try:
        # Check if the path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The path '{file_path}' does not exist.")

        # Check if the path is a file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The path '{file_path}' is not a file.")

        # Open the file
        seq_list = []
        with open(file_path, 'r') as file:
            for line in file:
                seq_list.append(line.strip())

    except FileNotFoundError as e:
        # Handle the case where the file or path does not exist
        return f"Error: {e}"
    except Exception as e:
        # Handle other exceptions
        return f"An error occurred: {e}"
    return seq_list


class Triplet:
    def __init__(self, seq1, seq2, seq3):
        """
        Might add 1 more dict {A:num_A, T:num_T, G:num_G, C:num_C} for prob of exist - ongoing
        :param seq1:
        :param seq2:
        :param seq3:
        """
        self.sequences = [seq1, seq2, seq3]
        self.length = len(seq1)  # Assume all sequences have the same length

    def consensus(self):
        transposed_sequences = zip(*self.sequences)

        consensus_sequence = ''

        for pos in transposed_sequences:
            # Count the occurrences of each character at the current position
            char_counts = Counter(pos)
            # print(char_counts.most_common(1)[0])
            most_common_char = char_counts.most_common(1)[0][0]
            consensus_sequence += most_common_char

        return consensus_sequence


class Edge:

    def __init__(self, pairs):
        """
        :param pairs: 2 vertices
        """
        self.pairs = pairs
        self.edge_w = 0

    def hamming_distance(self):
        if len(self.pairs[0]) != len(self.pairs[1]):
            raise ValueError("Input sequences must have equal length")  # Calculate Hamming distance
        self.edge_w = sum(c1 != c2 for c1, c2 in zip(self.pairs[0], self.pairs[1]))
        return self.edge_w


if __name__ == '__main__':
    ROOT = os.getcwd()
    file_list = os.listdir(ROOT)
    input_seqs = read_input('sequences.inp')
    # print(input_seqs)
    terminals = input_seqs[:5]
    int_nodes = []
    for seqs in combinations(terminals, 3):
        print(seqs)
        int_nodes.append(Triplet(seqs[0], seqs[1], seqs[2]).consensus())

    # Remove duplicates
    print(f'Original internal nodes:{len(int_nodes)}')
    print(int_nodes)
    int_nodes = list(set(int_nodes))
    print(f'Generated {len(int_nodes)} internal nodes as:')
    print(int_nodes)
    v_e_list = []
    for pairs in combinations(terminals + int_nodes, 2):
        # print(len())
        pairs_w = list(pairs) + [Edge(pairs).hamming_distance()]
        v_e_list.append(pairs_w)
    print(v_e_list)

    # ans = getAns(v_e_list=v_e_list,
    #        seqList=terminals + int_nodes,
    #        terminals=terminals)
    #
    # print("\n\n\n------------------------")
    #
    # print(ans)


    # # Graph
    # g = nx.DiGraph()
    # for i in v_e_list:
    #     [u, v, w] = i
    #     g.add_edge(u, v, weight=w)
    # print(g)
