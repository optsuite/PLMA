import ctypes
import torch
import time
import os

class QAPBackendGPU:
    """GPU backend that wraps qap_solver_batch.so for batch QAP operations."""
    def __init__(self, device='cuda'):
        self.lib = None
        self.device = torch.device(device)
        self.rand_states_ptr = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(script_dir, 'qap_solver_batch.so')
        self._load_library(so_path)
        self._define_interfaces()

    def _load_library(self, so_path):
        try:
            if not os.path.exists(so_path):
                raise FileNotFoundError(f"Library file not found: {so_path}. Please compile first.")
            self.lib = ctypes.cdll.LoadLibrary(so_path)
        
        except (OSError, FileNotFoundError) as e:
            print(f"Failed to load backend library {so_path}. Error: {e}")
            raise e

    def _define_interfaces(self):
        """Define interfaces for all batch C++ functions."""
        self.lib.create_curand_states_batch.argtypes = [ctypes.c_longlong, ctypes.c_ulonglong]
        self.lib.create_curand_states_batch.restype = ctypes.c_void_p
        
        self.lib.destroy_curand_states_batch.argtypes = [ctypes.c_void_p]

        self.lib.compute_cost_batch_qap_cuda.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int
        ]

        self.lib.mcmc_step_batch_cuda.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.local_search_batch_qap_cuda.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
        ]
        
    def setup_rand_states(self, num_states):
        """Initialize cuRAND states for random number generation."""
        if self.rand_states_ptr:
            self.lib.destroy_curand_states_batch(self.rand_states_ptr)
        seed = int(time.time())
        self.rand_states_ptr = self.lib.create_curand_states_batch(ctypes.c_longlong(num_states), seed)

    def __del__(self):
        if self.rand_states_ptr:
            self.lib.destroy_curand_states_batch(self.rand_states_ptr)

    def compute_cost(self, p_samples_batch: torch.Tensor, D_batch: torch.Tensor, F_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute QAP costs for batch permutations.
        
        Args:
            p_samples_batch: Permutations of shape (num_instances, num_samples, n).
            D_batch: Distance matrices of shape (num_instances, n, n).
            F_batch: Flow matrices of shape (num_instances, n, n).
            
        Returns:
            Costs of shape (num_instances, num_samples).
        """
        num_instances, num_samples, n = p_samples_batch.shape
        
        p_flat = p_samples_batch.reshape(-1, n).contiguous().to(torch.int32)
        D = D_batch.contiguous().to(torch.double)
        F = F_batch.contiguous().to(torch.double)
        
        costs = torch.zeros(num_instances * num_samples, dtype=torch.float64, device=self.device)
        
        self.lib.compute_cost_batch_qap_cuda(
            n, F.data_ptr(), D.data_ptr(), p_flat.data_ptr(), costs.data_ptr(),
            num_instances, num_samples
        )
        return costs.reshape(num_instances, num_samples)

    def mcmc_step(self, p_samples_batch: torch.Tensor, heatmap: torch.Tensor, mcmc_steps: int):
        """
        Perform MCMC steps on batch permutations guided by heatmap.
        
        Args:
            p_samples_batch: Permutations of shape (num_instances, num_samples, n).
            heatmap: Guidance heatmap of shape (n, n).
            mcmc_steps: Number of MCMC steps to perform.
            
        Returns:
            Updated permutations of shape (num_instances, num_samples, n).
        """
        if not self.rand_states_ptr:
            raise RuntimeError("cuRAND states not initialized. Please call setup_rand_states first.")
        
        num_instances, num_samples, n = p_samples_batch.shape
        
        p_flat = p_samples_batch.reshape(-1, n).contiguous().to(torch.int32)
        h_batch = heatmap.contiguous().to(torch.double)
        
        self.lib.mcmc_step_batch_cuda(
            n, p_flat.data_ptr(), h_batch.data_ptr(), self.rand_states_ptr,
            num_instances, num_samples, mcmc_steps
        )
        return p_flat.reshape(num_instances, num_samples, n)
        
    def local_search(self, p_samples_batch: torch.Tensor, D_batch: torch.Tensor, F_batch: torch.Tensor, num_actions: int, max_iter: int):
        """
        Perform local search on batch permutations.
        
        Args:
            p_samples_batch: Permutations of shape (num_instances, num_samples, n).
            D_batch: Distance matrices of shape (num_instances, n, n).
            F_batch: Flow matrices of shape (num_instances, n, n).
            num_actions: Number of random actions to try per iteration.
            max_iter: Maximum number of iterations.
            
        Returns:
            Improved permutations of shape (num_instances, num_samples, n).
        """
        if not self.rand_states_ptr:
            raise RuntimeError("cuRAND states not initialized. Please call setup_rand_states first.")

        num_instances, num_samples, n = p_samples_batch.shape
        p_flat = p_samples_batch.reshape(-1, n).contiguous().to(torch.int32)
        D = D_batch.contiguous().to(torch.double)
        F = F_batch.contiguous().to(torch.double)

        self.lib.local_search_batch_qap_cuda(
            p_flat.data_ptr(), D.data_ptr(), F.data_ptr(), n,
            num_instances, num_samples, num_actions, max_iter, self.rand_states_ptr
        )
        return p_flat.reshape(num_instances, num_samples, n)

