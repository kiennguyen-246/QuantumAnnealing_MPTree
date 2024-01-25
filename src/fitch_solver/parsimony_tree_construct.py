import io
import os
from collections import Counter
from itertools import combinations

from Bio import Phylo
from fitch_solver import is_optimized


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


class Triplet:
    def __init__(self, seq1, seq2, seq3):
        self.sequences = [seq1, seq2, seq3]
        self.length = len(seq1)

    def consensus(self):
        transposed_sequences = zip(*self.sequences)
        consensus_sequence = ''

        for pos in transposed_sequences:
            char_counts = Counter(pos)
            most_common_char = char_counts.most_common(1)[0][0]
            consensus_sequence += most_common_char

        return consensus_sequence


class Edge:
    def __init__(self, pairs):
        self.pairs = pairs
        self.edge_w = 0

    def hamming_distance(self):
        if len(self.pairs[0]) != len(self.pairs[1]):
            raise ValueError("Input sequences must have equal length")
        self.edge_w = sum(c1 != c2 for c1, c2 in zip(self.pairs[0], self.pairs[1]))
        return self.edge_w


class TreeReader:
    @staticmethod
    def read_tree_file(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Remove leading and trailing whitespaces
                line = line.strip()
                if line:
                    seq1, seq2, dist = line.split()
                    data.append((seq1, seq2, int(dist)))
        return data


class ParsimonyNode:
    def __init__(self, label, parent=None):
        self.label = label
        self.parent = parent
        self.children = []

    def add_child(self, child, distance):
        child_node = ParsimonyNode(child, self)
        self.children.append((child_node, distance))
        return child_node


class ParsimonyTree:
    def __init__(self):
        self.nodes = {}

    def add_edge(self, parent_label, child_label, distance):
        parent_node = self.nodes.get(parent_label, ParsimonyNode(parent_label))
        child_node = parent_node.add_child(child_label, distance)

        # Update nodes dictionary
        self.nodes[parent_label] = parent_node
        self.nodes[child_label] = child_node

    def get_root(self):
        for node in self.nodes.values():
            if node.parent is None:
                return node


class FitchSolver:
    def __init__(self, parsimony_tree):
        self.parsimony_tree = parsimony_tree
        self.solutions = {}

    def solve(self, node=None):
        if node is None:
            node = self.parsimony_tree.get_root()

        if not node.children:  # If it's a leaf node
            return {node.label: 0}

        # If the solutions for this node have been already computed, return it
        if node.label in self.solutions:
            return self.solutions[node.label]

        # Recursively solve for the children
        children = node.children

        if len(children) == 1:
            child, distance = children[0]
            child_solution = self.solve(child)

            # Cache the result for future use
            self.solutions[node.label] = child_solution
            return child_solution

        left_child, left_distance = children[0]
        right_child, right_distance = children[1]

        left_solution = self.solve(left_child)
        right_solution = self.solve(right_child)

        # Calculate the intersection and union of bases
        intersection_bases = set(left_solution.keys()) & set(right_solution.keys())
        union_bases = set(left_solution.keys()) | set(right_solution.keys())

        # Calculate the new solution
        if intersection_bases:
            result = {base: 0 for base in intersection_bases}
            cost = left_solution.get(list(left_solution.keys())[0], 0) + right_solution.get(list(right_solution.keys())[0], 0)
        else:
            result = {base: 1 for base in union_bases}
            cost = left_solution.get(list(left_solution.keys())[0], 0) + right_solution.get(list(right_solution.keys())[0], 0) + 1

        # Cache the result for future use
        self.solutions[node.label] = result

        return result


def print_tree_dfs(inp_root):
    stack = [(inp_root, 0)]

    while stack:
        current_node, level = stack.pop()
        print("  " * level + f"{current_node.label}")

        for child, distance in reversed(current_node.children):
            stack.append((child, level + 1))
            print("  " * (level + 1) + f"Child: {child.label}, Distance: {distance}")


def print_tree(root, level=0, prefix="Root: "):
    if root is not None:
        if level == 0:
            print(prefix + root.label)
        else:
            print(" " * (level * 4) + prefix + root.label)

        for i, (child, distance) in enumerate(root.children):
            if i == len(root.children) - 1:
                print_tree(child, level + 1, "└── ")
            else:
                print_tree(child, level + 1, "├── ")


def to_newick(node):
    if not node.children:
        return node.label

    children_newick = ",".join([f"{to_newick(child[0])}:{child[1]}" for child in node.children])
    return f"({children_newick}){node.label}"


if __name__ == '__main__':
    ROOT = os.getcwd()
    file_list = os.listdir(ROOT)
    input_seqs = SequenceReader.read_input('../../quantum_tree_output/6terminals_1.txt')
    terminals = input_seqs[:5]
    int_nodes = []

    for seqs in combinations(terminals, 3):
        print(seqs)
        int_nodes.append(Triplet(seqs[0], seqs[1], seqs[2]).consensus())

    print(f'Terminal list:')
    print(terminals)

    int_nodes = list(set(int_nodes))
    print(f'Generated {len(int_nodes)} internal nodes as:')
    print(int_nodes)

    tree_edges = TreeReader.read_tree_file(
        "../../quantum_tree_output/6terminals_1.txt")
    print(f"tree_edges: {tree_edges}")

    parsimony_tree = ParsimonyTree()

    for edge in tree_edges:
        root, child, distance = edge
        parsimony_tree.add_edge(root, child, int(distance))

    root = parsimony_tree.get_root()
    print(f"Root: {root.label}")

    print("Tree DFS traversing:")
    print_tree_dfs(root)

    print("Fitch solutions:")
    fitch_solver = FitchSolver(parsimony_tree)
    solutions = fitch_solver.solve(root)
    for label, cost in solutions.items():
        print(f"Node: {label}, Fitch cost: {cost}")
