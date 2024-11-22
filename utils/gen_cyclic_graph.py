import sys
import numpy as np
import random

# import numpy as np
import networkx as nx

def generate_cycle_graph_adjacency_matrix(n):
    # Create a cycle graph with n nodes
    G = nx.cycle_graph(n)
    
    # Get the adjacency matrix in numpy array format
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    
    return adjacency_matrix

def generate_b_vector(n):
    """ Generate a random b vector with one negative value that balances the sum to zero. """
    b = [0 for _ in range(n)]
    b[0] = -1
    b[n//2] = 1
    # sum_b = sum(b)
    # b.append(-sum_b)
    # random.shuffle(b)  # Shuffle to ensure the negative value is in a random position
    return b

def write_test_case_to_file(matrix, b, file_path="input.txt"):
    """ Write the generated graph and b vector to a file in the specified format. """
    
    n = len(matrix)
    
    with open(file_path, "w") as f:
        # Write the size of the matrix
        f.write(f"{n}\n")
        
        # Write the adjacency matrix
        for row in matrix:
            f.write(" ".join(map(str, map(int, row))) + "\n")
        
        # Write the b vector
        f.write(" ".join(map(str, b)) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 a.py <n>")
        sys.exit(1)

    n = int(sys.argv[1])
    sparsity = 0.6  # You can adjust this value or make it a command-line argument as well

    # Generate the graph
    matrix = generate_cycle_graph_adjacency_matrix(n)

    # Generate the b vector
    b = generate_b_vector(n)

    # Write the test case to a file
    write_test_case_to_file(matrix, b)

    print(f"Test case generated and written to test_case.txt")
