import matplotlib.pyplot as plt

# Load data from the file
x = []
y = []

with open('eta_graph.txt', 'r') as file:
    for line in file:
        values = line.split()
        if len(values) == 2:  # Ensure the line has exactly two elements
            x.append(int(values[0]))
            y.append(float(values[1]))

# Plotting the graph
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Graph of X vs Y')
plt.grid(True)
plt.show()
