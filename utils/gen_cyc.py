import sys

def generate_cyclic_csr(n):
    # Number of edges in a cycle graph is equal to the number of vertices
    num_vertices = n
    num_edges = n

    # Row pointer: For a cyclic graph, each node has 2 connections
    row_ptr = [2 * i for i in range(n+1)]
    row_ptr[-1] = 2 * n

    # Column offset: Cyclic connections (i to (i+1)%n and i to (i-1)%n)
    # col_offset = [(i-1) % n for i in range(n)] + [(i+1) % n for i in range(n)]
    col_offset = []
    for i in range(n):
        col_offset.append((i-1) % n)
        col_offset.append((i+1) % n)

    # Values: All values set to 1 (edges are unweighted)
    values = [1.0] * num_edges * 2

    # The b array: n zeros except for first and middle node
    b = [0] * n
    b[0] = -1
    middle = n // 2
    b[middle] = 1

    # Generate file
    filename = f"cyclic{n}"
    with open(filename, 'w') as f:
        # First line: number of vertices and number of edges
        f.write(f"{num_vertices} {num_edges * 2}\n")
        
        # Row pointer
        f.write(" ".join(map(str, row_ptr)) + "\n")

        # Column offset
        f.write(" ".join(map(str, col_offset)) + "\n")

        # Values (all 1s)
        f.write(" ".join(map(str, values)) + "\n")

        # b array (-1 at the first node, 1 at the middle node)
        f.write(" ".join(map(str, b)) + "\n")

    print(f"File {filename} generated successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 gen_cyc.py <number_of_nodes>")
        sys.exit(1)

    # Number of nodes
    n = int(sys.argv[1])

    # Generate the cyclic graph CSR file
    generate_cyclic_csr(n)
