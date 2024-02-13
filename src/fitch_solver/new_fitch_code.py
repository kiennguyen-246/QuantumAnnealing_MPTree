import os

from ete3 import Tree


class FitchScorer:
    class FitchNode:
        def __init__(self, node, node_list, idx):
            # Initialize FitchNode with information about the node's candidates
            if node.is_leaf():
                self.children = []
                self.candidates = {node.name[idx]}
            else:
                self.children = [child for child in node.children if node_list.get(child)]
                child_candidates = [node_list[child] for child in self.children]
                common_elements = set.intersection(*child_candidates) if child_candidates else set()
                if common_elements:
                    self.candidates = common_elements
                else:
                    self.candidates = set.union(*child_candidates) if child_candidates else set()

                # Update node_list with candidates for the current node
                node_list[node] = self.candidates

    def __init__(self, ete3_tree, idx=0):
        # Initialize FitchScorer with an ETE3 tree and an index for character states
        self.ete3_tree = ete3_tree
        self.idx = idx
        self.fitch_tree = None
        self.modified_tree = None

    def bottom_up_phase(self):
        # Perform the bottom-up phase of the Fitch algorithm
        root = self.ete3_tree.get_tree_root()
        node_list = {}
        self.fitch_tree = self.FitchNode(root, node_list, self.idx)

    def top_down_refinement(self):
        # Perform the top-down refinement of the Fitch algorithm
        if not self.fitch_tree:
            raise ValueError("Call bottom_up_phase before top_down_refinement.")

        root = self.ete3_tree.get_tree_root()
        root_candidates = self.fitch_tree.candidates

        # Arbitrarily assign one state to the root if there are multiple candidates
        if len(root_candidates) > 1:
            root.add_feature("state", next(iter(root_candidates)))

        def assign_state(node, state):
            # Recursively assign states to internal nodes
            if not node.is_leaf():
                for child in node.children:
                    if state in self.FitchNode(child, {}, self.idx).candidates:
                        assign_state(child, state)
                        break

        def refine(node):
            modified = False  # Initialize a flag to track modifications
            if not node.is_leaf() and len(node.children) == 2:
                for child in node.children:
                    child_state = self.FitchNode(child, {}, self.idx).candidates
                    if child_state and node.name not in child_state:
                        # Arbitrarily assign any state from S(v) to node v
                        assign_state(child, next(iter(child_state)))
                        modified = True  # Set the flag to True when a modification is made
                    modified |= refine(child)  # Recursive call and update the flag
            return modified

        # Clone the original tree to keep it unchanged
        self.modified_tree = self.ete3_tree.copy()

        # Perform the top-down refinement on the cloned tree
        refine(self.modified_tree.get_tree_root())

    def print_difference(self):
        # Print any differences in mutation points between the original and modified trees
        if not self.modified_tree:
            raise ValueError("Call top_down_refinement before print_difference.")

        for node1, node2 in zip(self.ete3_tree.traverse(), self.modified_tree.traverse()):
            if not node1.is_leaf() and not node2.is_leaf():
                children1 = [child.name for child in node1.children]
                children2 = [child.name for child in node2.children]
                if set(children1) != set(children2):
                    print(f"Difference in node {node1.name}: {children1} -> {children2}")


class Node:
    def __init__(self, label, distance=None):
        self.label = label
        self.distance = distance
        self.children = []


class NewickFormatter:
    def __init__(self, file_path):
        self.edges = self.read_edges(file_path)
        self.adjacency_list = {}
        self.root = None

    def read_edges(self, file_path):
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    edges.append((parts[0], parts[1], int(parts[2])))
        return edges

    def build_tree(self):
        # Step 1: Build an adjacency list
        for edge in self.edges:
            node1, node2, distance = edge
            if node1 not in self.adjacency_list:
                self.adjacency_list[node1] = []
            if node2 not in self.adjacency_list:
                self.adjacency_list[node2] = []
            self.adjacency_list[node1].append((node2, distance))
            self.adjacency_list[node2].append((node1, distance))

        # Step 2: Perform DFS to construct the Newick format tree
        def dfs(node_label, visited):
            if node_label in visited:
                return None
            visited.add(node_label)

            current_node = Node(node_label)

            for neighbor, distance in self.adjacency_list.get(node_label, []):
                child_node = dfs(neighbor, visited)
                if child_node:
                    child_node.distance = distance
                    current_node.children.append(child_node)

            return current_node

        # Find the root label
        root_label = self.edges[0][0]

        # Initialize visited set
        visited = set()

        # Build the Newick format tree
        self.root = dfs(root_label, visited)

        if not self.root:
            print("Error: The provided edge information contains a cycle.")
            return None

    def to_newick(self, node):
        newick = ""
        if node.children:
            newick += "(" + ",".join(self.to_newick(child) for child in node.children) + ")"
        newick += node.label
        if node.distance is not None:
            newick += f":{node.distance}"
        return newick

    def get_newick_tree(self):
        if not self.root:
            self.build_tree()

        return self.to_newick(self.root) + ";"


# Example Usage
for file in os.listdir('quantum_tree_output'):
    file_path = os.path.join('quantum_tree_output', file)
    formatter = NewickFormatter(file_path)
    newick_tree = formatter.get_newick_tree()

    if newick_tree:
        print(f"Newick format tree for file {file_path}")
        print(newick_tree)
        print('\n')

        input_string = newick_tree
        ete3_tree = Tree(input_string, format=1)
        fitch_scorer = FitchScorer(ete3_tree)
        fitch_scorer.bottom_up_phase()
        fitch_scorer.top_down_refinement()
        for node in ete3_tree.traverse():
            if node.is_leaf():
                print(f"Leaf: {node.name}, par: {node.up.name}")
            elif node.is_root():
                print(f"Root: {node.name}, children: {[child.name for child in node.children]}")
            else:
                print(f"internal: {node.name}, children: {[child.name for child in node.children]}")
        print("Original Tree:")
        print(ete3_tree)

        print("\nModified Tree:")
        print(fitch_scorer.modified_tree)

        print("\nDifferences in Mutation Points:")
        fitch_scorer.print_difference()
