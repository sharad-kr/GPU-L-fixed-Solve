### Brief Description of the Project: GPU-Based Fixed Packet Laplacian Solver

#### **Project Overview:**
This project is a GPU-based implementation of a Laplacian solver optimized to work with a fixed number of message packets, designed to improve runtime efficiency in solving large-scale graph-based Laplacian systems. By introducing modifications to the existing DRW-LSolve algorithm and implementing it in CUDA, this solution leverages the power of GPU parallelism to enhance the performance and scalability of the solver for real-world graphs.

---

#### **Technical Highlights:**

1. **Fixed Number of Packets:**
   - Unlike the original algorithm, which dynamically adjusts the number of packets, this implementation works with a predetermined, fixed number of message packets.
   - This optimization reduces unnecessary computations, leading to improved runtime performance while maintaining the algorithm's correctness.

2. **CUDA-Based Implementation:**
   - Utilized CUDA programming for highly parallelized execution of graph-based message-passing operations.
   - Kernels were written for:
     - Randomized neighbor selection using **cuRAND**.
     - Efficient queue updates and message passing using atomic operations.
     - Iterative updates for steady-state calculation of occupancy probabilities.

3. **Key Optimizations:**
   - **Efficient Message Handling:** 
     - Introduced a mechanism to handle packet movement between graph nodes while avoiding race conditions.
     - Used atomic operations to synchronize updates across threads.
   - **Memory Coalescing and Resource Utilization:**
     - Employed CSR (Compressed Sparse Row) format for graph representation to optimize memory access patterns.
     - Leveraged GPU global and shared memory effectively to handle message-passing computations.
   - **Convergence Checking:**
     - Implemented an iterative convergence mechanism to terminate computations based on changes in occupancy probabilities.

4. **GPU Skills Demonstrated:**
   - **Parallelization Expertise:**
     - Designed and implemented parallel kernels for message passing, queue updates, and random neighbor selection.
   - **Memory Management:**
     - Allocated and managed GPU memory for graph data structures (e.g., CSR format), message packets, and queues.
   - **Optimization with Libraries:**
     - Integrated CUDA libraries like **cuRAND** for random number generation and **Thrust** for sorting and reductions.

---

#### **Impact of the Work:**

1. **Improved Runtime:**
   - By fixing the number of message packets, the runtime was significantly reduced compared to dynamic packet implementations. The reduction in convergence iterations and packet-related overhead led to a more scalable solution.

2. **Scalability:**
   - Demonstrated the ability to handle large-scale graphs efficiently, showcasing improved performance for datasets with millions of nodes and edges.

3. **Accurate Results:**
   - The implementation ensured convergence to accurate steady-state solutions while maintaining numerical stability.

4. **Real-World Applicability:**
   - The optimized solver is well-suited for applications requiring efficient solutions to Laplacian systems in fields like graph-based machine learning, social network analysis, and physics simulations.

---



## Input File Format

The input file must be in CSR (Compressed Sparse Row) format and should follow this structure:

1. **First Line**: Two space-separated integers representing the number of nodes and edges.
2. **Second Line**: `row_ptr` array.
3. **Third Line**: `col_offset` array.
4. **Fourth Line**: `values` array (edge weights).
5. **Fifth Line**: Input `b` vector, where the sum of all values in this line is zero.

## Valid Command

To run the executable, use the following command:

`./a.out <file_path> <Number_of_message_packets> <Epoch> <output_path>`

- **file_path**: Path to the input file.
- **Number_of_message_packets**: Specifies the number of message packets.
- **Epoch**: Number of iterations to run.
- **output_path**: Path where the output will be saved.

## Output

After running the `main.cu` executable, the results will be stored at the specified `output_path`.

To visualize the graph and other details, run the following command:

`python3 plot.py <output_path>`

## Hyperparameters

1. Most hyperparameters are passed through the command line, as described in the usage instructions.
2. The weights for the weighted average calculation are hardcoded in the kernel `update_eta_by_qLen`.
