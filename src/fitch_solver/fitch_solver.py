class Node:
    def __init__(self, name=None, children=None):
        self.name = name
        self.children = children or []

def fitch_algorithm(node):
    if not node.children:
        # Leaf node
        return set(node.name)

    # Recursively apply Fitch algorithm to children
    child_states = [fitch_algorithm(child) for child in node.children]

    # Fitch algorithm logic
    common_states = set.intersection(*child_states)
    if not common_states:
        # No common state, choose one arbitrary state
        common_states = {list(child_states[0])[0]}

    return common_states

def construct_tree_from_newick(newick_string):
    stack = []
    current_node = None

    for char in newick_string:
        if char == '(':
            # Start a new internal node
            stack.append(Node())
            if current_node:
                stack[-2].children.append(current_node)
            current_node = None
        elif char == ')':
            # Finish the current internal node
            if current_node:
                stack[-1].children.append(current_node)
            current_node = stack.pop()
        elif char == ',':
            # Move to the next sibling
            if current_node:
                stack[-1].children.append(current_node)
            current_node = None
        elif char.isalnum():
            # Create a leaf node
            current_node = Node(name=char)

    return current_node

def is_optimized(newick_string):
    root = construct_tree_from_newick(newick_string)
    result = fitch_algorithm(root)
    return len(result) == 1

# Example usage
newick_tree = "((AGGAT:3,(CTGTA:3,(AACAT:1)AAGAT:3)ATGGC:2)ATCAC:1,(ATTAG:1)ATTAT:3)ATCGC"
print(is_optimized(newick_tree))
