import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, os.pardir))
sys.path.append(project_root)
import time
import pandas as pd
import numpy as np
import torch
import scipy.io
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from src.finetune import mcmc_finetune
from src.utils.utils import seed_everything

def _pick_one_file_per_instance(data_root):
    """
    Recursively scan data_root and select one file per instance.
    Prefer .mat files over .txt files when both exist.
    """
    all_files = []
    for root, _, files in os.walk(data_root):
        for name in files:
            path = os.path.join(root, name)
            if path.lower().endswith((".mat", ".txt")):
                all_files.append(path)
    all_files.sort()

    grouped = {}
    for path in all_files:
        base = os.path.splitext(os.path.basename(path))[0]
        if base == "2016NAMELIST":
            continue
        grouped.setdefault(base, []).append(path)

    selected = []
    for base in sorted(grouped):
        paths = grouped[base]
        mat_path = next((p for p in paths if p.lower().endswith(".mat")), None)
        selected.append(mat_path if mat_path is not None else paths[0])
    return selected


def _normalize_instance_name(name):
    normalized = name.replace("\\", "/").strip().lower()
    if normalized.endswith(".mat") or normalized.endswith(".txt"):
        return os.path.splitext(normalized)[0]
    return normalized


def _matches_instance_filter(problem_name, selected_instances):
    if not selected_instances:
        return True

    normalized_problem = _normalize_instance_name(problem_name)
    normalized_basename = _normalize_instance_name(os.path.basename(problem_name))

    for candidate in selected_instances:
        normalized_candidate = _normalize_instance_name(candidate)
        if normalized_candidate in {normalized_problem, normalized_basename}:
            return True
    return False


def _select_next_m(m_lb, m_ub, m_quantile):
    if m_ub - m_lb <= 1:
        return m_ub

    m = int(np.floor(m_lb + m_quantile * (m_ub - m_lb)))
    return min(max(m, m_lb + 1), m_ub - 1)


def _load_bm_matrix(file_path):
    """
    Load one BM instance matrix as np.ndarray(float32, [n, n]).
    """
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix == ".mat":
        mat = scipy.io.loadmat(file_path)
        if "A" in mat:
            matrix = np.asarray(mat["A"], dtype=np.float32)
        else:
            matrix = None
            for key, value in mat.items():
                if key.startswith("__"):
                    continue
                arr = np.asarray(value)
                if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                    matrix = arr.astype(np.float32)
                    break
            if matrix is None:
                raise ValueError("No square matrix found in .mat file")
    elif suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            raw = np.fromstring(f.read(), sep=" ", dtype=np.float32)
        if raw.size < 3:
            raise ValueError("Invalid txt format: too few tokens")
        n = int(raw[0])
        matrix_tokens = raw[2:]
        if matrix_tokens.size != n * n:
            raise ValueError(f"Invalid txt format: expected {n*n} values, got {matrix_tokens.size}")
        matrix = matrix_tokens.reshape(n, n)
    else:
        raise ValueError(f"Unsupported BM file type: {file_path}")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: {matrix.shape}")
    return (matrix != 0).astype(np.float32)


def load_bm(data_root, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), selected_instances=None):
    """
    Yield BM instances as (problem_name, (n, A_tensor)).
    """
    problem_files = _pick_one_file_per_instance(data_root)
    print(f"Found {len(problem_files)} BM instances in '{data_root}'.")
    for file_path in problem_files:
        try:
            dense_matrix = _load_bm_matrix(file_path)
            n = dense_matrix.shape[0]
            A = torch.tensor(dense_matrix, dtype=torch.float32, device=device)
            problem_name = os.path.relpath(file_path, data_root)
            if not _matches_instance_filter(problem_name, selected_instances):
                continue
            yield problem_name, (n, A)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")


def config_bm(n):
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # Finetuning parameters
        "num_finetune_steps": 50,
        "learning_rate": 1e-1,
        "random_seed_base": None,
        "random_seed_round": 0,
        "random_seed_stride": 1,
        "adaptive_step_extension": True,
        "adaptive_extension_window": 25,
        "adaptive_extension_extra_steps": 50,
        "adaptive_extension_best_cost_threshold": 40.0,
        # MCMC sampling parameters
        "num_starts": 20,
        "num_chains": 20,
        "chain_length": np.clip(n // 3, 10, 50),
        "local_search_iter": np.clip(n, 20, 200),
        "num_actions": np.clip(2 * n, 50, 200),
        "entropy_weight":  1 / n / np.log(n),
        "initial_samples": None

    }

    return config


def bandwidth(matrix):
    """
    Compute the bandwidth of a sparse matrix.
    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    
    rows, cols = matrix.nonzero()
    
    if rows.size == 0:
        return 0
        
    bandwidth = np.max(np.abs(rows - cols))
    return bandwidth

def driver_bm(
    dataset,
    output_dir,
    model_params=None,
    checkpoint=None,
    verbose=True,
    selected_instances=None,
    reuse_heatmap_offset=False,
    reuse_heatmap_renorm=False,
    reuse_heatmap_sinkhorn_iterations=1,
    reuse_logits_offset=False,
    reuse_samples_only=False,
    reuse_last_successful_net=True,
    reuse_last_successful_samples=False,
    m_quantile=0.5,
    adaptive_step_extension=True,
    adaptive_extension_window=25,
    adaptive_extension_extra_steps=50,
    adaptive_extension_best_cost_threshold=40.0,
    seed=2024,
    backend_seed_round_stride=1009,
    backend_seed_instance_stride=1000003,
):
    if model_params is None:
        raise ValueError("model_params is required for mcmc_finetune (used to instantiate the model).")

    os.makedirs(output_dir, exist_ok=True)
    results = []
    data_root = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
    dataset_tag = os.path.basename(os.path.normpath(data_root))
    output_path = os.path.join(output_dir, f"results_bm_{dataset_tag}.csv")
    if selected_instances:
        print("Selected instances:")
        for name in selected_instances:
            print(f"  - {name}")
    data_loader = load_bm(data_root=data_root, selected_instances=selected_instances)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for problem_idx, (problem_name, (n, A)) in enumerate(data_loader):
        print("-" * 80)
        print(f"Processing Instance: {problem_name} (n={n})")
        A_numpy = A.cpu().numpy()
        permutation = reverse_cuthill_mckee(csr_matrix(A_numpy))
        A_rcm = A_numpy[permutation, :][:, permutation]
        bw_rcm = bandwidth(A_rcm)
        m_lb, m_ub = 0, n - 1
        m = _select_next_m(m_lb, m_ub, m_quantile)
        i, j = torch.arange(n, device=device), torch.arange(n, device=device)
        B0 = torch.abs(i[:, None] - j).to(torch.float32)
        opt = 0.5
        k = 0
        start_time = time.time()

        config = config_bm(n)
        config["random_seed_base"] = int(seed) + problem_idx * int(backend_seed_instance_stride)
        config["random_seed_stride"] = int(backend_seed_round_stride)
        config["adaptive_step_extension"] = adaptive_step_extension
        config["adaptive_extension_window"] = adaptive_extension_window
        config["adaptive_extension_extra_steps"] = adaptive_extension_extra_steps
        config["adaptive_extension_best_cost_threshold"] = adaptive_extension_best_cost_threshold
        net = None
        warm_checkpoint = checkpoint
        previous_heatmap = None
        previous_logits = None
        last_success_net = None
        last_success_heatmap = None
        last_success_logits = None
        last_success_initial_samples = None
        while m_ub - m_lb > 1:
            k = k + 1
            Bm = torch.clamp(B0 - m, min=0.0)
            candidate_net = None
            candidate_heatmap = None
            candidate_logits = None
            candidate_initial_samples = None
            if Bm.max().item() <= 0:
                cost_value = 0.0
                run_time = 0.0
            else:
                config["random_seed_round"] = k - 1
                if reuse_last_successful_samples:
                    config["initial_samples"] = last_success_initial_samples
                if reuse_samples_only:
                    round_net = None
                    round_checkpoint = checkpoint
                    heatmap_source = None
                    logits_source = None
                elif reuse_last_successful_net:
                    round_net = last_success_net
                    round_checkpoint = checkpoint if last_success_net is None else None
                    heatmap_source = last_success_heatmap
                    logits_source = last_success_logits
                else:
                    round_net = net
                    round_checkpoint = warm_checkpoint
                    heatmap_source = previous_heatmap
                    logits_source = previous_logits
                heatmap_offset = None
                logits_offset = None
                if reuse_heatmap_offset and heatmap_source is not None and round_net is not None:
                    with torch.no_grad():
                        current_heatmap = round_net(A.unsqueeze(0), Bm.unsqueeze(0)).detach()
                    heatmap_offset = heatmap_source - current_heatmap
                if reuse_logits_offset and logits_source is not None and round_net is not None:
                    with torch.no_grad():
                        current_logits = round_net.forward_logits(A.unsqueeze(0), Bm.unsqueeze(0)).detach()
                    logits_offset = logits_source - current_logits

                _, cost, _, run_time, aux = mcmc_finetune(
                    A.unsqueeze(0),
                    Bm.unsqueeze(0),
                    config,
                    opt,
                    net=round_net,
                    model_params=model_params,
                    checkpoint=round_checkpoint,
                    verbose=verbose,
                    return_aux=True,
                    heatmap_offset=heatmap_offset,
                    heatmap_offset_renorm=reuse_heatmap_renorm,
                    heatmap_offset_sinkhorn_iterations=reuse_heatmap_sinkhorn_iterations,
                    logits_offset=logits_offset,
                )
                candidate_net = aux["net"]
                candidate_heatmap = aux["final_heatmap"]
                candidate_logits = aux["final_logits"]
                candidate_initial_samples = aux["initial_samples"]
                net = None if reuse_samples_only else candidate_net
                if not reuse_last_successful_samples:
                    config["initial_samples"] = candidate_initial_samples
                previous_heatmap = None if reuse_samples_only else candidate_heatmap
                previous_logits = None if reuse_samples_only else candidate_logits
                warm_checkpoint = checkpoint if reuse_samples_only else None
                cost_value = cost.item() if torch.is_tensor(cost) else float(cost)

            if cost_value < 1:
                m_ub = m
                if not reuse_samples_only and candidate_net is not None:
                    last_success_net = candidate_net
                    last_success_heatmap = candidate_heatmap
                    last_success_logits = candidate_logits
                if candidate_initial_samples is not None:
                    last_success_initial_samples = candidate_initial_samples
                    if reuse_last_successful_samples:
                        config["initial_samples"] = last_success_initial_samples
            else:
                m_lb = m

            print(f"iter={k}, m={m}, cost={cost_value:.4f}, runtime={run_time:.2f}s, m_lb={m_lb}, m_ub={m_ub}")
            if m_lb >= bw_rcm:
                print(f"Stop early: m_lb={m_lb} >= bw_rcm={bw_rcm}, no need to continue.")
                break
            m = _select_next_m(m_lb, m_ub, m_quantile)

        m = min(m, bw_rcm)
        run_time = time.time() - start_time
        print(f"Instance: {problem_name}, n={n}, RCM Bandwidth: {bw_rcm}, Final m: {m}, Time: {run_time:.2f}s")
        results.append({
            'problem_name': problem_name,
            'n': n,
            'bw_rcm': bw_rcm,
            'bw': m,
            'time': run_time
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, float_format='%.3f')
    print(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/bm/sparseMarket', help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default='./results/bm', help='Directory to save results')
    parser.add_argument('--repetitions', '-r', type=int, default=10, help='Number of repetitions per instance')
    parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose output')

    parser.add_argument("--init_dim", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_gcn_layers", type=int, default=10)
    parser.add_argument("--num_att_layers", type=int, default=1)
    parser.add_argument("--clipping_value", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--model_type", type=str, default="simplenet", choices=["simplenet"])
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to model checkpoint')
    parser.add_argument("--reuse_heatmap_offset", action="store_true", help="Reuse previous-round heatmap as an additive offset.")
    parser.add_argument("--reuse_heatmap_renorm", action="store_true", help="Re-normalize the offset heatmap with Sinkhorn after adding the offset.")
    parser.add_argument("--reuse_heatmap_sinkhorn_iterations", type=int, default=1, help="Sinkhorn iterations used when --reuse_heatmap_renorm is enabled.")
    parser.add_argument("--reuse_logits_offset", action="store_true", help="Reuse previous-round pre-Sinkhorn logits as an additive offset.")
    parser.add_argument("--reuse_samples_only", action="store_true", help="Reset the network each round and only reuse initial samples.")
    parser.add_argument("--reuse_last_successful_net", action="store_true", default=True, help="Reuse the most recent successful net instead of the previous round net.")
    parser.add_argument("--reuse_last_successful_samples", action="store_true", help="Reuse the most recent successful initial samples instead of the previous round samples.")
    parser.add_argument("--m_quantile", type=float, default=0.5, help="Choose the next m as a quantile in (m_lb, m_ub). 0.5 is standard binary search; values >0.5 bias toward larger m.")
    parser.add_argument("--seed", type=int, default=2024, help="Global random seed for reproducibility.")
    parser.add_argument("--backend_seed_round_stride", type=int, default=1009, help="Stride added to the backend seed for each outer BM iteration.")
    parser.add_argument("--backend_seed_instance_stride", type=int, default=1000003, help="Stride added to the backend seed for each benchmark instance.")
    parser.add_argument("--adaptive_step_extension", action="store_true", default=True, help="Extend finetune once if the recent BestCost trace is still improving and the current BestCost is already small.")
    parser.add_argument("--adaptive_extension_window", type=int, default=25, help="How many trailing BestCost values to inspect for recent improvement.")
    parser.add_argument("--adaptive_extension_extra_steps", type=int, default=50, help="Extra finetune steps added when adaptive extension triggers.")
    parser.add_argument("--adaptive_extension_best_cost_threshold", type=float, default=40.0, help="Adaptive extension only triggers when the current BestCost is below this threshold.")
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Only run selected BM instances. Match by relative path, basename, or stem.",
    )

    args = parser.parse_args()
    if args.reuse_heatmap_offset and args.reuse_logits_offset:
        raise ValueError("reuse_heatmap_offset and reuse_logits_offset cannot be enabled at the same time.")
    if args.reuse_samples_only and (args.reuse_heatmap_offset or args.reuse_logits_offset):
        raise ValueError("reuse_samples_only cannot be combined with heatmap/logits reuse.")
    if args.reuse_samples_only and args.reuse_last_successful_net:
        raise ValueError("reuse_samples_only cannot be combined with reuse_last_successful_net.")
    if not (0.0 < args.m_quantile < 1.0):
        raise ValueError("m_quantile must be in the open interval (0, 1).")
    if args.adaptive_extension_window <= 1:
        raise ValueError("adaptive_extension_window must be greater than 1.")
    if args.adaptive_extension_extra_steps < 0:
        raise ValueError("adaptive_extension_extra_steps must be non-negative.")
    if args.backend_seed_round_stride <= 0 or args.backend_seed_instance_stride <= 0:
        raise ValueError("backend seed strides must be positive.")

    seed_everything(args.seed)
    
    model_params = {
        "model_type": args.model_type,
        "init_dim": args.init_dim,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_gcn_layers": args.num_gcn_layers,
        "num_att_layers": args.num_att_layers,
        "num_iterations": args.num_iterations,
        "temperature": args.temperature,
        "clipping_value": args.clipping_value,
    }

    print("Model Parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print(f"Learning Rate: {config_bm(100)['learning_rate']}")
    print(f"Number of Finetune Steps: {config_bm(100)['num_finetune_steps']}")
    print(f"Reuse Heatmap Offset: {args.reuse_heatmap_offset}")
    print(f"Reuse Heatmap Renorm: {args.reuse_heatmap_renorm}")
    print(f"Reuse Heatmap Sinkhorn Iterations: {args.reuse_heatmap_sinkhorn_iterations}")
    print(f"Reuse Logits Offset: {args.reuse_logits_offset}")
    print(f"Reuse Samples Only: {args.reuse_samples_only}")
    print(f"Reuse Last Successful Net: {args.reuse_last_successful_net}")
    print(f"Reuse Last Successful Samples: {args.reuse_last_successful_samples}")
    print(f"m Quantile: {args.m_quantile}")
    print(f"Seed: {args.seed}")
    print(f"Backend Seed Round Stride: {args.backend_seed_round_stride}")
    print(f"Backend Seed Instance Stride: {args.backend_seed_instance_stride}")
    print(f"Adaptive Step Extension: {args.adaptive_step_extension}")
    print(f"Adaptive Extension Window: {args.adaptive_extension_window}")
    print(f"Adaptive Extension Extra Steps: {args.adaptive_extension_extra_steps}")
    print(f"Adaptive Extension BestCost Threshold: {args.adaptive_extension_best_cost_threshold}")
    driver_bm(
        args.dataset,
        args.output_dir,
        model_params,
        checkpoint=args.checkpoint,
        verbose=args.verbose,
        selected_instances=args.instances,
        reuse_heatmap_offset=args.reuse_heatmap_offset,
        reuse_heatmap_renorm=args.reuse_heatmap_renorm,
        reuse_heatmap_sinkhorn_iterations=args.reuse_heatmap_sinkhorn_iterations,
        reuse_logits_offset=args.reuse_logits_offset,
        reuse_samples_only=args.reuse_samples_only,
        reuse_last_successful_net=args.reuse_last_successful_net,
        reuse_last_successful_samples=args.reuse_last_successful_samples,
        m_quantile=args.m_quantile,
        seed=args.seed,
        backend_seed_round_stride=args.backend_seed_round_stride,
        backend_seed_instance_stride=args.backend_seed_instance_stride,
        adaptive_step_extension=args.adaptive_step_extension,
        adaptive_extension_window=args.adaptive_extension_window,
        adaptive_extension_extra_steps=args.adaptive_extension_extra_steps,
        adaptive_extension_best_cost_threshold=args.adaptive_extension_best_cost_threshold,
    )
