import sys
import numpy as np
import random

def generate_connected_graph(n, sparsity=0.7):
    """ Generate a random n x n adjacency matrix for a graph.
        sparsity controls the density of the graph.
        Lower sparsity means more edges. """
    
    matrix = np.zeros((n, n))

    # Ensure graph connectivity by making sure every node has at least one edge
    for i in range(n):
        while np.sum(matrix[i]) == 0:
            for j in range(i + 1, n):
                if random.random() > sparsity:
                    matrix[i][j] = 1
                    matrix[j][i] = 1

    # Add additional random edges based on sparsity
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() > sparsity:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix

def generate_b_vector(n):
    """ Generate a random b vector with one negative value that balances the sum to zero. """
    b = [random.uniform(0.1, 2) for _ in range(n-1)]
    sum_b = sum(b)
    b.append(-sum_b)
    random.shuffle(b)  # Shuffle to ensure the negative value is in a random position
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
    matrix = generate_connected_graph(n, sparsity)

    # Generate the b vector
    b = generate_b_vector(n)

    # Write the test case to a file
    write_test_case_to_file(matrix, b)

    print(f"Test case generated and written to test_case.txt")
