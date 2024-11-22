import matplotlib.pyplot as plt

def parse_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Extract metadata
    vertices = int(lines[0].split(":")[1].strip())
    source_nodes = int(lines[1].split(":")[1].strip())
    sink_nodes = int(lines[2].split(":")[1].strip())
    message_packets = int(lines[3].split(":")[1].strip())
    epoch = int(lines[4].split(":")[1].strip())
    error = float(lines[5].split(":")[1].strip())
    time_taken = int(lines[6].split(":")[1].strip().split()[0])
    
    # Extract the data points
    data = []
    for line in lines[7:]:
        if line.strip():  # Check if the line is not empty
            parts = line.split()
            if len(parts) == 2:  # Check if there are exactly two parts (index and value)
                index, value = map(float, parts)
                data.append((index, value))

    return vertices, source_nodes, sink_nodes, message_packets, epoch, error, time_taken, data

def plot_graph(vertices, source_nodes, sink_nodes, message_packets, epoch, error, time_taken, data):
    # Unzip the data into two lists: indices and values
    indices, values = zip(*data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, values, marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.title(f"Graph type : Cyclic\nEpoch : {epoch}\n#Nodes: {vertices} | #Source Nodes: {source_nodes} | #Sink Nodes: {sink_nodes} | "
              f"#Message Packets: {message_packets} | Error: {error:.6f} | Time Taken: {time_taken} ms")
    plt.xlabel("Node Index")
    plt.ylabel("Eta Value")

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Replace 'output.txt' with the path to your file
    filename = 'eta_graph.txt'
    
    # Parse the file
    vertices, source_nodes, sink_nodes, message_packets, epoch, error, time_taken, data = parse_file(filename)
    
    # Plot the graph
    plot_graph(vertices, source_nodes, sink_nodes, message_packets, epoch, error, time_taken, data)
