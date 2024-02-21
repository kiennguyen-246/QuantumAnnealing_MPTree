from collections import Counter
import os
from itertools import combinations
import networkx as nx
from makeGraph import getAns
import numpy as np

class SequenceReader:
    @staticmethod
    def read_input(file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The path '{file_path}' does not exist.")

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The path '{file_path}' is not a file.")

            seq_list = []
            with open(file_path, 'r') as file:
                for line in file:
                    seq_list.append(line.strip())

        except FileNotFoundError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An error occurred: {e}"
        return seq_list

    @staticmethod
    def read_input_phy(file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The path '{file_path}' does not exist.")

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"The path '{file_path}' is not a file.")

            seq_list = []
            with open(file_path, 'r') as file:
                lines = file.readlines()

                for line in lines[1:]:
                    # Start from line 1, skipping line 0
                    line = line.strip() # \n
                    line = line.split() # \t -> 2 string [0]: species, [1]: dna
                    seq_list.append(line[1])

        except FileNotFoundError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"An error occurred: {e}"
        return seq_list

    @staticmethod
    def statistic_module(seq_list):
        seq_len = len(seq_list[0])  # Assuming all sequences in seq_list have the same length
        stat = []

        for idx in range(seq_len):
            accu = {}
            counter = {'A': 0, 'T': 0, 'C': 0, 'G': 0, '-': 0}

            for seq in seq_list:
                if seq[idx] in {'-', 'M', 'H', 'Y', 'R', 'S', 'W', 'K', 'B', 'D', 'V', 'N', '?'}:
                    counter['-'] += 1
                else:
                    for item in ('A', 'T', 'C', 'G'):
                        counter[item] += 1 if seq[idx] == item else 0

            for item in ('A', 'T', 'C', 'G'):
                accu[item] = counter[item] / (seq_len - counter['-']) if (seq_len - counter['-']) != 0 else 0

            stat.append(accu)

        return stat


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

        for idx, pos in enumerate(transposed_sequences):
            # Count the occurrences of each character at the current position
            char_counts = Counter(pos)
            # print(char_counts.most_common(1))
            most_common_char = char_counts.most_common(1)[0][0]
            print(f"Most common character at pos{idx}: {most_common_char}")
            consensus_sequence += most_common_char

        return consensus_sequence

    def consensus_2(self, stat):
        # Calculate statistics using the statistic_module method
        # stat = Triplet.statistic_module(self.sequences)

        transposed_sequences = zip(*self.sequences)
        consensus_sequence = ''
        idx = 0
        for pos, pos_stat in zip(transposed_sequences, stat):
            idx+=1
            # Count the occurrences of each character at the current position (skip '-')
            char_counts = Counter(filter(lambda x: x not in {'-', 'M', 'H', 'Y', 'R', 'S', 'W', 'K', 'B', 'D', 'V', 'N', '?'}, pos))

            # If there is a tie, use the stat to select the highest accuracy nucleotide
            if char_counts:
                most_common_char = max(char_counts, key=lambda x: (char_counts[x], pos_stat[x]))
            else:
                most_common_char = '-'

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
    # New input for phylo file
    ROOT = os.getcwd()
    file_list = os.listdir(ROOT)
    # input_seqs = SequenceReader.read_input('sequences.inp')
    for directory in os.listdir(ROOT + '/data_treebase'):
    # for i in range (0, 10000000000):
        if ".phy" not in directory:
            continue
        if directory < "dna_M1110_330_1711.phy":
            continue
        # if directory[5] not in {'1', '2'}:
        #     continue
        print(directory)
        os.environ['PHYLO_FILE'] = directory
        input_seqs = SequenceReader.read_input_phy('data_treebase/' + directory)

        # os.environ['PHYLO_FILE'] = "sequences"
        # input_seqs = SequenceReader.read_input('sequences.inp')
        stat = SequenceReader.statistic_module(input_seqs)
        # print(input_seqs)
        terminals = input_seqs[:4]
        for idx, term in enumerate(terminals):
            print(f"{idx}) {term}")
        int_nodes = []
        for seqs in combinations(terminals, 3):
            # print(seqs)
            int_nodes.append(Triplet(seqs[0], seqs[1], seqs[2]).consensus_2(stat))

        # Remove duplicates
        print(f'Original internal nodes:{len(int_nodes)}')
        # print(int_nodes)
        int_nodes = list(set(int_nodes))
        print(f'Generated {len(int_nodes)} internal nodes as:')
        for idx, node in enumerate(int_nodes):
            print(f"{idx}) {node}")

        v_e_list = []
        for pairs in combinations(terminals + int_nodes, 2):
            # print(len())
            pairs_w = list(pairs) + [Edge(pairs).hamming_distance()]
            v_e_list.append(pairs_w)
        print(v_e_list)

        ans = getAns(v_e_list=v_e_list,
               seqList=terminals + int_nodes,
               terminals=terminals)

        print("\n\n\n------------------------")

        print(ans)

        tree_output_directory = ROOT + '/../quantum_tree_output/'
        if not os.path.exists(tree_output_directory):
            os.makedirs(tree_output_directory)
        with open(tree_output_directory + os.getenv("PHYLO_FILE") + ".txt", "w") as f:
            for edge in ans:
                f.write(str(edge[0]) + " " + str(edge[1]) + " " + str(edge[2]) + "\n")
        sum_ans = np.sum([i[2] for i in ans])
        # if sum_ans != 10:
        #     raise ValueError("Sum of edges is not 10")


    # # Graph
    # g = nx.DiGraph()
    # for i in v_e_list:
    #     [u, v, w] = i
    #     g.add_edge(u, v, weight=w)
    # print(g)