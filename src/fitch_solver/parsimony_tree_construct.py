import os


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
        print(f"Newick format tree for file {file_path}\n")
        print(newick_tree)
        print('\n')
