#include <curand_kernel.h>
#include <stdio.h>
#include<chrono>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/iterator/constant_iterator.h>

struct message {
    int msg_idx;
    int node;
};

int read_file_into_csr(const char* file_path, int** row_ptr, int** col_offset, float** values, float** b) {
    FILE* graph = fopen(file_path, "r");
    if (graph == NULL) {
        fprintf(stderr, "Unable to open the file at %s\n", file_path);
        return -1;
    }

    int V;
    fscanf(graph, "%d", &V);
    int non_zero_values = 0;

    float degree[V];
    memset(degree, 0, sizeof(degree));

    float** matrix = (float**)malloc(sizeof(float*) * V);
    for (int i = 0; i < V; i++) {
        matrix[i] = (float*)malloc(sizeof(float) * V);
        memset(matrix[i], 0, sizeof(float) * V);  // Initialize the matrix to zero
    }

    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            fscanf(graph, "%f", &matrix[i][j]);
            if (matrix[i][j] != 0) {
                non_zero_values++;
                degree[i] += matrix[i][j];
            }
        }
    }

    // Normalize the weights
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (matrix[i][j] > 0) {
                matrix[i][j] = matrix[i][j] / degree[i];
            }
        }
    }

    *row_ptr = (int*)malloc(sizeof(int) * (V + 1));
    *col_offset = (int*)malloc(sizeof(int) * non_zero_values);
    *values = (float*)malloc(sizeof(float) * non_zero_values);
    *b = (float*)malloc(sizeof(float) * V);

    int r = 0;
    int c = 0;
    int v = 0;

    for (int i = 0; i < V; i++) {
        (*row_ptr)[r] = c;
        for (int j = 0; j < V; j++) {
            if (matrix[i][j] != 0) {
                (*col_offset)[c] = j;
                (*values)[v] = matrix[i][j];
                c++;
                v++;
            }
        }
        r++;
    }
    (*row_ptr)[r] = c;
    float sum = 0;
    for (int i = 0; i < V; i++) {
        fscanf(graph, "%f", &((*b)[i]));
        sum += (*b)[i];

    }


    for (int i = 0; i < V; i++) {
        free(matrix[i]);
    }
    free(matrix);

    fclose(graph);

    return V;
}


int read_csr(const char* file_path, int** row_ptr, int** col_ptr, float** values, float** b) {
    FILE *file = fopen(file_path, "r");
    if (file == nullptr) {
        perror("Failed to open file");
        return -1; 
    }

    int V, E;
    if (fscanf(file, "%d", &V) != 1 || fscanf(file, "%d", &E) != 1) {
        perror("Failed to read V or E");
        fclose(file);
        return -1; 
    }

    *row_ptr = (int*)malloc(sizeof(int) * (V + 1));
    *col_ptr = (int*)malloc(sizeof(int) * E);
    *values = (float*)malloc(sizeof(int) * E);
    *b = (float*)malloc(sizeof(float) * V);

    if (*row_ptr == nullptr || *col_ptr == nullptr || *values == nullptr || *b == nullptr) {
        perror("Failed to allocate memory");
        fclose(file);
        return -1; 
    }

    for (int i = 0; i <= V; ++i) {
        if (fscanf(file, "%d", &(*row_ptr)[i]) != 1) {
            perror("Failed to read row_ptr");
            fclose(file);
            return -1; 
        }
    }

    for (int i = 0; i < E; ++i) {
        if (fscanf(file, "%d", &(*col_ptr)[i]) != 1) {
            perror("Failed to read col_ptr");
            fclose(file);
            return -1; 
        }
    }

    for (int i = 0; i < E; ++i) {
        if (fscanf(file, "%f", &(*values)[i]) != 1) {
            perror("Failed to read values");
            fclose(file);
            return -1; 
        }
    }

    for (int i = 0; i < V; ++i) {
        if (fscanf(file, "%f", &(*b)[i]) != 1) {
            perror("Failed to read b");
            fclose(file);
            return -1; 
        }
    }


    for(int i = 0 ; i<V+1 ; i++){
        int start_idx = (*row_ptr)[i];
        int end_idx = (*row_ptr)[i+1];
        float sum = 0;
        for(int j = start_idx; j< end_idx ; j++){
            sum += (*values)[j];
        }
        for(int j = start_idx; j< end_idx ; j++){
            (*values)[j] =  (*values)[j]/sum;
        }
    }

    fclose(file);
    return V;
}


int b2J(int V, float* b, float* neg_value, int* neg_index, int &source_nodes, int &sink_nodes) {
    *neg_value = 0;
    for (int i = 0; i < V; i++) {
        if (b[i] < 0) {
            *neg_value = b[i];
            *neg_index = i;
            sink_nodes++;
        }
    }
    *neg_value = -(*neg_value);
    for (int i = 0; i < V; i++) {
        if (b[i] > 0) {
            b[i] = b[i] / *neg_value;
            source_nodes++;
        }
    }
    return 0;
}

void print_csr(int V, int* row_ptr, int* col_offset, float* values) {
    for (int i = 0; i <= V; i++) {
        printf("%d ", row_ptr[i]);
    }
    printf("\n");
    for (int i = 0; i < V; i++) {
        int curr = row_ptr[i];
        int end = row_ptr[i + 1];
        if (end - 1 < curr) continue;
        for (int j = curr; j < end; j++) {
            printf("%d ", col_offset[j]);
        }
    }
    printf("\n");

    for (int i = 0; i < V; i++) {
        int curr = row_ptr[i];
        int end = row_ptr[i + 1];
        if (end - 1 < curr) continue;
        for (int j = curr; j < end; j++) {
            printf("%f ", values[j]);
        }
    }
    printf("\n");
    
    
}


__global__ void flush_boolArray(bool *d_recv, int V){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= V) return;
    d_recv[idx] = 0;

}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ int get_source_node(curandState *state, int *d_source_idx, int source_nodes, int V) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    float random_number = curand_uniform(&localState);  // Generate random float in [0, 1]
    state[idx] = localState;  // Store the state back
    int lower_bound = 0;
    int upper_bound = source_nodes;
    int node_idx = lower_bound + (int)(random_number * (upper_bound - lower_bound));
    int node = d_source_idx[node_idx];
    return node;
}

__device__ int get_nbr(curandState *state, int* row_ptr, int* col_offset, float* values, int node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = state[idx];
    float random_number = curand_uniform(&localState);  // Generate random float in [0, 1]
    state[idx] = localState;  // Store the state back
    int lower_bound = row_ptr[node];
    int upper_bound = row_ptr[node+1];
    int nbr_node_idx = lower_bound + (int)(random_number * (upper_bound - lower_bound));
    int nbr_node = col_offset[nbr_node_idx];

    return nbr_node;
}

__global__ void processMessages(int *d_row_ptr, int *d_col_offset, float *d_values, struct message *d_messages,
                                int num_msg_packets, int V,float *d_b ,int *d_source_idx, int source_nodes, 
                                curandState *state, bool* d_recv, int neg_index, int *d_traveller) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // print("hy")
    if (idx < num_msg_packets) {
        struct message *msg = &d_messages[idx];
        int curr_node = msg->node;

        if (d_b[curr_node] < 0) {
            int src_node = get_source_node(state, d_source_idx, source_nodes,V);  // Device-compatible random number generation
            msg->node = src_node;
            d_traveller[idx] = src_node;
            d_recv[src_node] = 1;
        } else {
            int nbr = get_nbr(state, d_row_ptr, d_col_offset, d_values, curr_node);  // Device-compatible random number generation
            msg->node = nbr;
            
            d_recv[nbr] = 1;
            d_traveller[idx] = nbr;
        }
    }
}


__global__ void update_eta_by_qLen(float* d_eta, float* d_prev_eta ,int *d_cnt, int V, bool *d_recv, int epoch, int num_msg_packets) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_prev_eta[idx] = d_eta[idx];
    d_eta[idx] = ((epoch-1)*d_eta[idx] + (1*d_cnt[idx]))/epoch;
    d_recv[idx] = 0;
    d_cnt[idx] = 0;

}

__global__ void normalise(float *d_eta, int V, int epoch){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= V) return;
    d_eta[idx] = d_eta[idx]/epoch;
}


__global__ void update_qlen(int *d_traveller, int num_msg_packets, int *d_cnt){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= num_msg_packets) return;

    int node = d_traveller[idx];
    atomicAdd(&d_cnt[node], 1);
}





int main(int argc, char** argv) {
    // print("working")
    auto start = std::chrono::high_resolution_clock::now();
    if (argc != 5) {
        fprintf(stderr, "Valid command: %s <file_path> <Number of message-packets> <Epoch> <output_path>\n", argv[0]);
        return -1;
    }

    const char* output_path = argv[4];

    const char* file_path = argv[1];
    int num_msg_packets = atoi(argv[2]);
    int max_epoch = atoi(argv[3]);
    int *h_row_ptr = NULL, *h_col_offset = NULL, *h_negIndex = NULL, *h_cnt = NULL, *h_source_idx = NULL, *h_traveller = NULL;
    int source_nodes = 0;
    int sink_nodes = 0;

    float *h_values = NULL, *h_b = NULL, *h_negValue = NULL;

    h_negValue = (float*)malloc(sizeof(float));
    h_negIndex = (int*)malloc(sizeof(int));

    
    struct message *h_messages = (struct message*)malloc(sizeof(struct message) * num_msg_packets);
    h_traveller = (int*)malloc(sizeof(int) * num_msg_packets);


    int V = read_csr(file_path, &h_row_ptr, &h_col_offset, &h_values, &h_b);

    h_cnt = (int*)calloc(V,sizeof(int));
    // h_queue_size = (int*)calloc(V, sizeof(int));

    int msg_alloc = 0;
    bool* h_recv = (bool*)calloc(V,sizeof(bool));
    for (int i = 0; i < V; i++) {
        if (h_b[i] <= 0) continue;
        int num_msges = round(h_b[i] * num_msg_packets);
        // printf(" %f %d |",b[i],num_msges);
        int end = msg_alloc + num_msges;
        for (int curr = msg_alloc; curr < end  && curr < num_msg_packets; curr++) {
            h_messages[curr].msg_idx = curr;
            h_messages[curr].node = i;
            h_recv[i] = 1;
            h_traveller[curr] = i;
            // h_queue_size[i]++;
            msg_alloc++;
        }
    }
    for(int i = 0 ; i<V ; i++){
        if(h_recv[i]){
            h_cnt[i]++;
        }
    }

    b2J(V, h_b, h_negValue, h_negIndex, source_nodes, sink_nodes);
    h_source_idx = (int*)malloc(sizeof(int) * source_nodes);
    int w = 0;
    for(int i = 0 ; i < V ;i++){
        if(h_b[i] > 0){
            h_source_idx[w] = i;
            w++;
        }
    }
    
    for(int i = 0 ; i<source_nodes ; i++){
    	printf("%d\n",h_source_idx[i]);
    }

    printf("Number of packets in network : %d\n",msg_alloc);
    printf("Negative index : %d\n", *h_negIndex);
    if (V == -1) {
        return -1;
    }
    float *h_eta = (float*)calloc(V,sizeof(float));
    float* h_prev_eta = (float*)calloc(V, sizeof(float));


    int *d_row_ptr, *d_col_offset, *d_queue_sizes, *d_cnt, *d_source_idx, *d_traveller;
    float *d_values, *d_b, *d_eta, *d_prev_eta;
    bool *d_recv;
    double* d_err;
    struct message *d_messages;
    curandState *d_state;
    
    // Allocate device memory
    cudaMalloc(&d_source_idx, sizeof(int)* source_nodes);
    cudaMalloc(&d_row_ptr, sizeof(int) * (V + 1));
    cudaMalloc(&d_col_offset, sizeof(int) * h_row_ptr[V]);
    cudaMalloc(&d_values, sizeof(float) * h_row_ptr[V]);
    cudaMalloc(&d_b, sizeof(float) * V);
    cudaMalloc(&d_queue_sizes, sizeof(int) * V);
    cudaMalloc(&d_cnt, sizeof(int) * V);
    cudaMalloc(&d_eta, sizeof(float) * V);
    cudaMalloc(&d_prev_eta, sizeof(float) * V);
    cudaMalloc(&d_messages, sizeof(struct message) * num_msg_packets);
    cudaMalloc(&d_state, sizeof(curandState) * num_msg_packets);  // Allocate memory for RNG states
    cudaMalloc(&d_recv, sizeof(bool) * V);
    cudaMalloc(&d_err, sizeof(double));
    cudaMalloc(&d_traveller, sizeof(int) * num_msg_packets);
    // cudaMalloc(&d_queue_size, sizeof(int) * V);

    cudaMemset(d_recv, 0, sizeof(bool) * V);
    cudaMemset(d_cnt, 0, sizeof(int) * V);
    cudaMemset(d_eta, 0, sizeof(float) * V);
    cudaMemset(d_prev_eta, 0, sizeof(float) * V);
    cudaMemset(d_traveller, 0, sizeof(int) * num_msg_packets);
    // Copy data from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_offset, h_col_offset, sizeof(int) * h_row_ptr[V], cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, sizeof(float) * h_row_ptr[V], cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, h_messages, sizeof(struct message) * num_msg_packets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, h_cnt, sizeof(int) * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_idx, h_source_idx, sizeof(int) * source_nodes, cudaMemcpyHostToDevice);


    int blockSize = 256;
    int msgBlocks = (num_msg_packets + blockSize - 1) / blockSize;
    int nodeBlocks = (V + blockSize - 1) / blockSize;

    // Setup the RNG states
    setup_kernel<<<msgBlocks, blockSize>>>(d_state, time(NULL));
    cudaDeviceSynchronize();
    
    flush_boolArray<<<nodeBlocks, blockSize>>>(d_recv, V);
    cudaDeviceSynchronize();
    // printf("grebgugb");
    int epoch = 1;
    while (epoch < max_epoch) {
        // epoch = epoch + num_msg_packets;
        // print("working")
        processMessages<<<msgBlocks, blockSize>>>(d_row_ptr, d_col_offset, d_values, d_messages, num_msg_packets, V,d_b ,d_source_idx,
                                                  source_nodes ,d_state, d_recv, *h_negIndex, d_traveller);
        cudaDeviceSynchronize();
        // set_qlen2zero<<<nodeBlocks, blockSize>>>(d_cnt, V);
        // cudaDeviceSynchronize();
        update_qlen<<<msgBlocks, blockSize>>>(d_traveller, num_msg_packets, d_cnt);
        cudaDeviceSynchronize();
        update_eta_by_qLen<<<nodeBlocks, blockSize>>>(d_eta, d_prev_eta, d_cnt, V, d_recv, epoch, num_msg_packets);
        cudaDeviceSynchronize();
        epoch++;
    }

    // normalise<<<nodeBlocks, blockSize>>>(d_eta, V, max_epoch);

    // Copy results back to host if needed
    cudaMemcpy(h_eta, d_eta, sizeof(float) * V, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prev_eta, d_prev_eta, sizeof(float) * V, cudaMemcpyDeviceToHost);
    float err = 0;
    for(int i  =  1 ; i < V ; i++){
        err = max(err, abs(h_eta[i] - h_prev_eta[i])/h_prev_eta[i]);
    }
    printf("Max eta_error = %f\n", err);
    FILE* eta_graph = fopen(output_path,"w");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    printf("Elapsed time: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());


    fprintf(eta_graph,"#Vertices: %d\n#Source nodes: %d\n#Sink nodes: %d\n#Message Packets: %d\n#Epoch : %d\nError: %f\nTime taken: %ld ms\n", V, source_nodes,sink_nodes,num_msg_packets, max_epoch ,err, std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
    

    printf("Eta distribution: \n");
    for (int i = 0; i < V; i++) {
        printf("eta[%d] : %f \n", i, h_eta[i]);
        fprintf(eta_graph, "%d %f\n", i, h_eta[i]);
    }
    printf("\n");
    fclose(eta_graph);

    cudaFree(d_row_ptr);
    cudaFree(d_col_offset);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_queue_sizes);
    cudaFree(d_cnt);
    cudaFree(d_eta);
    cudaFree(d_prev_eta);
    cudaFree(d_messages);
    cudaFree(d_state);
    cudaFree(d_recv);
    cudaFree(d_err);

   
    free(h_row_ptr);
    free(h_col_offset);
    free(h_values);
    free(h_b);
    free(h_negIndex);
    free(h_cnt);
    free(h_negValue);
    free(h_eta);
    free(h_prev_eta);
    free(h_recv);
    free(h_messages);

    return 0;

}
