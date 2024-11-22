import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import sys

# Function to read the graph from a text file
def read_csr_graph(file_path):
    with open(file_path, 'r') as f:
        # Reading the number of vertices and edges
        num_vertices, num_edges = map(int, f.readline().strip().split())
        
        # Reading row_ptr, col_offset, and values
        row_ptr = list(map(int, f.readline().strip().split()))
        col_offset = list(map(int, f.readline().strip().split()))
        values = list(map(float, f.readline().strip().split()))

    # Construct CSR matrix from the data
    csr_graph = csr_matrix((values, col_offset, row_ptr), shape=(num_vertices, num_vertices))

    return csr_graph

# Function to convert CSR matrix to NetworkX graph and print details
def process_graph(csr_graph):
    # Convert the CSR matrix to a NetworkX graph (use the updated function)
    G = nx.from_scipy_sparse_array(csr_graph, edge_attribute='weight')

    # Print basic graph details
    print(f"\nGraph information:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # 1. Graph Density
    density = nx.density(G)
    print(f"Density: {density:.4f}")

    # 2. Check if the graph is connected and calculate diameter if connected
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"Diameter: {diameter}")
    else:
        print("Graph is not connected, cannot compute diameter.")

    average_node_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(f"Average degree of nodes: {average_node_degree:.2f}")


    max_degree = max(dict(G.degree()).values())
    min_degree = min(dict(G.degree()).values())
    print(f"Maximum degree: {max_degree}")
    print(f"Minimum degree: {min_degree}")

# Main function
if __name__ == "__main__":
    # Check if a filename is passed as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python3 detail.py <filename>")
        sys.exit(1)

    # Get the filename from the command-line argument
    file_path = sys.argv[1]

    # Read the CSR graph from the file
    csr_graph = read_csr_graph(file_path)

    # Process the graph and print details using NetworkX
    process_graph(csr_graph)
