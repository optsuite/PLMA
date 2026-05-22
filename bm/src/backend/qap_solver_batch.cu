#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

// ===================================================================
// CUDA error checking macro
// ===================================================================
#define CUDA_CHECK(call)                                                 \
do {                                                                     \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)


// ===================================================================
// CUDA Kernels
// ===================================================================

/**
 * @brief Kernel: Compute total cost for a batch of solutions (batch version)
 * Each thread block is responsible for computing the cost of one solution sample.
 */
__global__ void compute_cost_batch_kernel(
    int n,
    const double* F_batch_d,
    const double* D_batch_d,
    const int* p_samples_batch_d,
    double* costs_d,
    int num_instances,
    int num_samples)
{
    extern __shared__ double sdata[];

    long long global_sample_idx = blockIdx.x;
    long long total_samples = (long long)num_instances * num_samples;
    if (global_sample_idx >= total_samples) return;

    int instance_idx = global_sample_idx / num_samples;

    const double* current_F = F_batch_d + (long long)instance_idx * n * n;
    const double* current_D = D_batch_d + (long long)instance_idx * n * n;
    const int* p = p_samples_batch_d + global_sample_idx * n;

    int tid_in_block = threadIdx.x;
    double cost_private = 0.0;
    for (int i = tid_in_block; i < n; i += blockDim.x) {
        for (int j = 0; j < n; ++j) {
            cost_private += current_D[i * n + j] * current_F[p[i] * n + p[j]];
        }
    }
    sdata[tid_in_block] = cost_private;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            sdata[tid_in_block] += sdata[tid_in_block + s];
        }
        __syncthreads();
    }

    if (tid_in_block == 0) {
        costs_d[global_sample_idx] = sdata[0];
    }
}


/**
 * @brief Kernel: Perform one MCMC transition step (batch version)
 * Each thread is responsible for one MCMC step for one solution sample.
 */
__global__ void mcmc_step_batch_kernel(
    int n,
    int* p_samples_batch_d,
    const double* heatmap_batch_d,
    curandState* rand_states_d,
    int num_instances,
    int num_samples)
{
    long long global_sample_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_samples = (long long)num_instances * num_samples;
    if (global_sample_idx >= total_samples) return;

    int instance_idx = global_sample_idx / num_samples;

    const double* current_heatmap = heatmap_batch_d + (long long)instance_idx * n * n;
    int* p = p_samples_batch_d + global_sample_idx * n;

    curandState local_state = rand_states_d[global_sample_idx];

    int r = curand(&local_state) % n;
    int s = (r + (curand(&local_state) % (n - 1)) + 1) % n;

    int pi_r = p[r];
    int pi_s = p[s];

    double current_prob = current_heatmap[r * n + pi_r] * current_heatmap[s * n + pi_s];
    double proposed_prob = current_heatmap[r * n + pi_s] * current_heatmap[s * n + pi_r];
    double ratio = proposed_prob / (current_prob + 1e-12);

    if (curand_uniform_double(&local_state) < ratio) {
        p[r] = pi_s;
        p[s] = pi_r;
    }

    rand_states_d[global_sample_idx] = local_state;
}


/**
 * @brief Kernel: Compute cost change (delta) for all atomic tasks based on flattened mapping strategy.
 */
__global__ void compute_delta_batch_kernel(
    int n,
    const double* F_batch_d,
    const double* D_batch_d,
    const int* p_samples_batch_d,
    const int* actions_d,
    double* deltas_d,
    int num_instances,
    int num_samples,
    int num_actions)
{
    long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_global_actions = (long long)num_instances * num_samples * num_actions;
    if (global_idx >= total_global_actions) return;

    long long total_actions_per_instance = (long long)num_samples * num_actions;
    int instance_idx = global_idx / total_actions_per_instance;
    long long local_idx_in_instance = global_idx % total_actions_per_instance;
    int sample_idx_in_instance = local_idx_in_instance / num_actions;

    const double* current_F = F_batch_d + (long long)instance_idx * n * n;
    const double* current_D = D_batch_d + (long long)instance_idx * n * n;
    long long p_sample_offset = (long long)instance_idx * num_samples + sample_idx_in_instance;
    const int* p = p_samples_batch_d + p_sample_offset * n;

    int r = actions_d[global_idx * 2 + 0];
    int s = actions_d[global_idx * 2 + 1];

    if (r == s) {
        deltas_d[global_idx] = 0.0;
        return;
    }

    int pi_r = p[r];
    int pi_s = p[s];
    double delta = 0.0;

    for (int k = 0; k < n; ++k) {
        if (k != r && k != s) {
            int pi_k = p[k];
            delta += (current_D[k * n + r] - current_D[k * n + s]) * (current_F[pi_k * n + pi_s] - current_F[pi_k * n + pi_r]);
            delta += (current_D[r * n + k] - current_D[s * n + k]) * (current_F[pi_s * n + pi_k] - current_F[pi_r * n + pi_k]);
        }
    }
    
    delta += (current_D[r*n+r] - current_D[s*n+s]) * (current_F[pi_s*n+pi_s] - current_F[pi_r*n+pi_r]);
    delta += (current_D[r*n+s] - current_D[s*n+r]) * (current_F[pi_s*n+pi_r] - current_F[pi_r*n+pi_s]);
    deltas_d[global_idx] = delta;
}


/**
 * @brief Kernel: Find the best 2-swap action with maximum cost reduction for each solution sample and apply it.
 */
__global__ void find_and_apply_best_action_batch_kernel(
    int n,
    int* p_samples_batch_d,
    const double* all_deltas_d,
    const int* all_actions_d,
    int num_instances,
    int num_samples,
    int num_actions)
{
    extern __shared__ char s_mem[];
    volatile double* s_deltas = (volatile double*)s_mem;
    volatile int* s_indices = (volatile int*)(s_mem + blockDim.x * sizeof(double));
    
    long long global_sample_idx = blockIdx.x;
    long long total_samples = (long long)num_instances * num_samples;
    if (global_sample_idx >= total_samples) return;

    int tid_in_block = threadIdx.x;
    long long action_offset = global_sample_idx * num_actions;
    const double* deltas = all_deltas_d + action_offset;

    double min_delta_private = 0.0;
    int best_action_idx_private = -1;
    for (int i = tid_in_block; i < num_actions; i += blockDim.x) {
        if (deltas[i] < min_delta_private) {
            min_delta_private = deltas[i];
            best_action_idx_private = i;
        }
    }
    s_deltas[tid_in_block] = min_delta_private;
    s_indices[tid_in_block] = best_action_idx_private;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid_in_block < s) {
            if (s_deltas[tid_in_block + s] < s_deltas[tid_in_block]) {
                s_deltas[tid_in_block] = s_deltas[tid_in_block + s];
                s_indices[tid_in_block] = s_indices[tid_in_block + s];
            }
        }
        __syncthreads();
    }

    if (tid_in_block == 0 && s_deltas[0] < 0.0) {
        int best_action_idx = s_indices[0];
        
        int instance_idx = global_sample_idx / num_samples;
        int sample_idx_in_instance = global_sample_idx % num_samples;
        long long p_sample_offset = (long long)instance_idx * num_samples + sample_idx_in_instance;
        int* p = p_samples_batch_d + p_sample_offset * n;
        
        const int* actions = all_actions_d + action_offset * 2;
        int r = actions[best_action_idx * 2 + 0];
        int s = actions[best_action_idx * 2 + 1];
        
        int temp = p[r];
        p[r] = p[s];
        p[s] = temp;
    }
}


/**
 * @brief Kernel: Generate random 2-swap actions for all atomic tasks.
 */
__global__ void generate_actions_batch_kernel(
    int n,
    int* actions,
    curandState* states,
    long long total_global_actions)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_global_actions) return;

    curandState local_state = states[idx];
    int r = curand(&local_state) % n;
    int s = (r + (curand(&local_state) % (n - 1)) + 1) % n;

    actions[idx * 2 + 0] = r;
    actions[idx * 2 + 1] = s;
    states[idx] = local_state;
}


/**
 * @brief Kernel: Initialize cuRAND states.
 */
__global__ void setup_rand_states_batch_kernel(curandState* states, unsigned long long seed, long long n_states) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    curand_init(seed, idx, 0, &states[idx]);
}


// ===================================================================
// C-style Host Functions (External Interface)
// ===================================================================
extern "C" {

/**
 * @brief Create and initialize cuRAND state array.
 */
curandState* create_curand_states_batch(long long num_states, unsigned long long seed) {
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, num_states * sizeof(curandState)));
    
    int threads_per_block = 256;
    long long blocks = (num_states + threads_per_block - 1) / threads_per_block;
    setup_rand_states_batch_kernel<<<blocks, threads_per_block>>>(d_states, seed, num_states);
    CUDA_CHECK(cudaGetLastError());
    return d_states;
}

/**
 * @brief Destroy cuRAND state array and free device memory.
 */
void destroy_curand_states_batch(curandState* states) {
    if (states) {
        CUDA_CHECK(cudaFree(states));
    }
}


/**
 * @brief Compute costs for all samples of all instances on GPU.
 */
void compute_cost_batch_qap_cuda(int n, const double* F_batch_d, const double* D_batch_d,
                                 const int* p_samples_batch_d, double* costs_d,
                                 int num_instances, int num_samples)
{
    long long total_samples = (long long)num_instances * num_samples;
    int threads_per_block = 256;
    dim3 grid_dim(total_samples);
    size_t shmem_size = threads_per_block * sizeof(double);

    compute_cost_batch_kernel<<<grid_dim, threads_per_block, shmem_size>>>(
        n, F_batch_d, D_batch_d, p_samples_batch_d, costs_d,
        num_instances, num_samples);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Perform MCMC steps for all samples of all instances on GPU.
 */
void mcmc_step_batch_cuda(int n, int* p_samples_batch_d, const double* heatmap_batch_d,
                          curandState* rand_states_d, int num_instances, int num_samples,
                          int mcmc_steps)
{
    long long total_samples = (long long)num_instances * num_samples;
    int threads_per_block = 256;
    long long blocks_in_grid = (total_samples + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < mcmc_steps; ++i) {
        mcmc_step_batch_kernel<<<blocks_in_grid, threads_per_block>>>(
            n, p_samples_batch_d, heatmap_batch_d, rand_states_d,
            num_instances, num_samples);
        CUDA_CHECK(cudaGetLastError());
    }
}


/**
 * @brief Perform one step of random neighborhood search on GPU.
 */
void local_search_batch_qap_cuda(
    int* p_samples_batch_d,
    const double* D_batch_d,
    const double* F_batch_d,
    int n,
    int num_instances,
    int num_samples,
    int num_actions,
    int max_iter,
    curandState* rand_states_d)
{
    long long total_global_actions = (long long)num_instances * num_samples * num_actions;
    long long total_samples = (long long)num_instances * num_samples;
    
    int* d_candidate_actions;
    double* d_all_deltas;
    CUDA_CHECK(cudaMalloc(&d_candidate_actions, total_global_actions * 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_deltas, total_global_actions * sizeof(double)));

    int threads_per_block = 256;
    long long blocks_for_actions_grid = (total_global_actions + threads_per_block - 1) / threads_per_block;
    dim3 blocks_for_samples_grid(total_samples); 
    size_t shmem_size = threads_per_block * sizeof(double) + threads_per_block * sizeof(int);

    for (int iter = 0; iter < max_iter; ++iter) {
        generate_actions_batch_kernel<<<blocks_for_actions_grid, threads_per_block>>>(
            n, d_candidate_actions, rand_states_d, total_global_actions);
        CUDA_CHECK(cudaGetLastError());
        
        compute_delta_batch_kernel<<<blocks_for_actions_grid, threads_per_block>>>(
            n, F_batch_d, D_batch_d, p_samples_batch_d, d_candidate_actions,
            d_all_deltas, num_instances, num_samples, num_actions);
        CUDA_CHECK(cudaGetLastError());
        
        find_and_apply_best_action_batch_kernel<<<blocks_for_samples_grid, threads_per_block, shmem_size>>>(
            n, p_samples_batch_d, d_all_deltas, d_candidate_actions,
            num_instances, num_samples, num_actions);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaFree(d_candidate_actions));
    CUDA_CHECK(cudaFree(d_all_deltas));
}

} // extern "C"