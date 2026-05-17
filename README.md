# PLMA

This is the code repository for PLMA (Permutation Learning with MCMC-based Adaptation), a learning framework for solving the Quadratic Assignment Problem (QAP), which seeks an optimal permutation $\pi \in \Pi_n$ minimizing the total assignment cost:

$$\min_{\pi \in \Pi_n} \sum_{i=1}^{n}\sum_{j=1}^{n} F_{ij}\, D_{\pi(i)\pi(j)}.$$

Here $F \in \mathbb{R}^{n\times n}$ denotes the flow matrix with entries $F_{ij}$, $D \in \mathbb{R}^{n\times n}$ denotes the distance matrix with entries $D_{kl}$, and $\Pi_n$ denotes the set of all permutations over $[n]$.

PLMA features:
- An efficient warm-started MCMC finetuning procedure
- An additive energy-based model (EBM) enabling $O(1)$-time 2-swap Metropolis-Hastings sampling
- A scalable cross-graph attention mechanism for modeling QAP interactions

## Setup

### Compilation of CUDA Kernels to Python Shared Library
The requisite high-performance kernel functions, contained within the CUDA source files (`.cu`), must be compiled into a dynamically loadable shared object (`.so`) file.
```
cd src/backend
# The exact module name may vary based on your environment configuration.
module load cuda 
nvcc -Xcompiler -fPIC -shared -o qap_solver_batch.so qap_solver_batch.cu -lcublas -lcurand
```
### Python Environment Initialization and Dependency Management
The python package dependencies are specified in the file `requirements.txt`. For the most convenient installation of the environment, we highly recommend using conda.

```
conda create -n plma python=3.10
conda activate plma
pip install -r requirements.txt
```
## Training
Specify all training configurations in the YAML files found within the `configs` directory. To train a model, pass the path to its corresponding configuration file.

For instance, to train on the uniformly random dataset ($n=100$), execute:
```
python train.py --config uniform100.yaml
```
## Evaluation
In our paper, we have evaluated PLMA on two families of synthetic QAP instances, and on the real-world QAPLIB and Taixxeyy benchmarks. Here we show how to run PLMA on these datasets and present partial results. 

### Running Scripts

```
./scripts/run_sawt.sh
./scripts/run_uniform.sh
./scripts/run_qaplib.sh
./scripts/run_tai.sh
```
### Results
We compare our model against a wide spectrum of established and modern approaches, categorized as follows.
- **Search-based solvers:** Robust Tabu Search (Ro-TS), Breakout Memetic Algorithm (BMA), Connolly's Simulated Annealing (C-SA);
- **Heuristic solvers:** Classic iterative algorithms commonly applied to graph matching, including IPFP, SM, and RRWM;
- **Learning-based solvers:** Two representative deep learning approaches, SAWT and NGM.

Results on two synthetic datasets are presented in the next two tables. Some crucial conclusions can be drawn:
1. The pretrained model (PLMA, $T=1$) offers an instant, high-quality solution that already outperforms other learning-based approaches;
2. With brief fine-tuning (PLMA, $T=50$), the model rapidly converges to near-optimality, surpassing all other baselines than Ro-TS (5k);
3. With 200 fine-tuning steps, PLMA consistently matches or surpasses the strong Ro-TS (5k) baseline in quality while being remarkably more efficient. 
#### Geometrically Structured Dataset
|  |  | $n=50$| |  | $n=100$ | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algorithm**| **Cost** | **Gap** | **Time** | **Cost** | **Gap** | **Time** |
| Ro-TS (1k) | 375.99 | 0.14% | 4m35s | 1593.27 | 0.13% | 38m56s |
| Ro-TS (5k) | **375.48** | **0.00%** | 22m53s | 1591.25 | 0.00% | 3h15m |
| BMA | 375.60 | 0.03% | 15m53s | 1591.58 | 0.02% | 2h21m |
| C-SA | 376.56 | 0.29% | 21m32s | 1592.95 | 0.11% | 2h45m |
| --- | --- | --- | --- | --- | --- | --- |
| IPFP | 378.76 | 0.88% | 11.47s | 1600.27 | 0.57% | 1m34s |
| IPFP (10) | 376.60 | 0.30% | 2m30s | 1594.76 | 0.22% | 17m34s |
| RRWM | 428.78 | 14.14% | 39.23s | 1700.33 | 6.86% | 6m32s |
| SM | 426.92 | 13.70% | 7.14s | 1753.10 | 10.17% | 1m40s |
| --- | --- | --- | --- | --- | --- | --- |
| NGM | 429.69 | 14.46% | 1m16s | 1773.71 | 11.47% | 2m29s |
| SAWT (10k) | 380.92 | 1.45% | 5m36s | 1617.30 | 1.64% | 10m43s |
| --- | --- | --- | --- | --- | --- | --- |
| **PLMA ($T=1$)** | 379.79 | 1.15% | 0.41s | 1607.84 | 1.04% | 4.27s |
| **PLMA ($T=50$)** | 375.55 | 0.20% | 19.88s | 1591.73 | 0.03% | 3m30s |
| **PLMA ($T=200$)** | **375.48** | **0.00%** | 1m19s | **1591.23** | **0.00%** | 13m58s |

#### Uniformly Random Dataset
|  |  | $n=50$| |  | $n=100$ | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algorithm**| **Cost** | **Gap** | **Time** | **Cost** | **Gap** | **Time** |
| Ro-TS (1k) | 523.08 | 0.22% | 4m36s | 2195.98 | 0.13% | 38m59s |
| Ro-TS (5k) | 521.91 | 0.00% | 22m59s | 2193.16 | 0.00% | 3h15m |
| BMA | 521.79 | -0.02% | 17m26s | 2194.14 | 0.04% | 3h32m |
| C-SA | 523.19 | 0.25% | 22m21s | 2193.69 | 0.02% | 2h45m |
| --- | --- | --- | --- | --- | --- | --- |
| IPFP | 530.74 | 1.69% | 7.45s | 2211.38 | 0.83% | 41.95s |
| IPFP (25) | 526.96 | 0.97% | 4m20s | 2203.29 | 0.46% | 19m20s |
| RRWM | 592.50 | 13.54% | 31.91s | 2432.34 | 10.91% | 5m6s |
| SM | 605.08 | 15.95% | 5.65s | 2457.47 | 12.05% | 1m23s |
| --- | --- | --- | --- | --- | --- | --- |
| NGM | 594.99 | 14.01% | 1m17s | 2438.52 | 11.19% | 2m29s |
| --- | --- | --- | --- | --- | --- | --- |
| **PLMA ($T=1$)** | 538.42 | 3.17% | 0.40s | 2243.43 | 2.29% | 4.32s |
| **PLMA ($T=50$)** | 523.95 | 0.40% | 19.51s | 2200.40 | 0.34% | 3m30s |
| **PLMA ($T=200$)** | **521.75** | **-0.03%** | 1m18s | **2193.13** | **0.00%** | 14m1s |

#### QAPLIB Benchmark
The following table presents the average performance of different algorithms on the QAPLIB dataset, categorized by instance class. The metrics reported include the average optimality Gap and the computation Time (in seconds). PLMA achieves the best overall performance, attaining the lowest average optimality gap of 0.06% while also requiring the least average computation time among all heuristic baselines.
|  | **Ro-TS** | | **SAWT** | | **IPFP** | | **PLMA** | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|**Class** | **Gap** | **Time** | **Gap** | **Time** | **Gap** | **Time** | **Gap** | **Time** |
| bur (26) | **0.00%** | 0.12 | 3.95% | 14.67 | 0.05% | 0.33 | **0.00%** | 0.08 |
| chr (12-25) | 0.48% | 0.16 | 147.54% | 14.18 | 14.90% | 0.26 | **0.00%** | 0.31 |
| els (19) | **0.00%** | 0.02 | 47.37% | 14.24 | 10.18% | 0.36 | **0.00%** | 0.09 |
| esc (16-128) | **0.00%** | 0.62 | 43.29% | 15.07 | 0.39% | 0.63 | **0.00%** | 0.07 |
| had (12-20) | **0.00%** | 0.00 | 5.17% | 14.23 | 0.08% | 0.39 | **0.00%** | 0.05 |
| kra (30-32) | **0.00%** | 0.24 | 32.92% | 14.77 | 0.65% | 0.62 | **0.00%** | 0.31 |
| lipa (20-90) | 0.03% | 3.28 | 1.40% | 17.32 | 1.07% | 4.93 | 0.08% | 1.57 |
| nug (12-30) | **0.00%** | 0.02 | 19.25% | 14.39 | 0.05% | 0.38 | **0.00%** | 0.17 |
| rou (12-20) | **0.00%** | 0.04 | 15.09% | 14.25 | 0.77% | 0.33 | **0.00%** | 0.08 |
| scr (12-20) | **0.00%** | 0.01 | 33.92% | 14.22 | 1.24% | 0.28 | **0.00%** | 0.08 |
| sko (42-100) | 0.05% | 29.22 | 16.17% | 19.06 | 0.30% | 8.40 | **0.03%** | 7.01 |
| ste (36) | 0.01% | 0.53 | 107.95% | 14.95 | 1.81% | 0.70 | **0.00%** | 0.55 |
| tai (10-256) | 0.23% | 25.16 | 34.67% | 16.62 | 0.92% | 3.33 | **0.20%** | 4.73 |
| tho (30-150) | **0.04%** | 38.74 | 24.05% | 17.19 | 0.42% | 12.46 | **0.04%** | 6.42 |
| wil (50-100) | **0.02%** | 26.06 | 9.52% | 17.64 | 0.12% | 9.66 | **0.02%** | 6.82 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Average** | 0.11% | 9.68 | 37.82% | 15.85 | 2.15% | 2.73 | **0.06%** | 2.16 |

#### Taixxeyy Instances
The next table presents the grouped results on Taixxeyy instances, with all metrics averaged over 10 independent runs. For each class, the reported mean and [min, max] gaps (%) are averages of per-instance statistics. PLMA delivers a low average gap of 2.56% and a reliable worst-case performance with an average maximum gap of 3.84%. This is in stark contrast to Ro-TS's highly erratic 81.01% average gap and catastrophic failures with a maximum gap of 285.64%.
| | **Ro-TS** | | | **PLMA** | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class** | **mean** | **[min, max]** | **Time** | **mean** | **[min, max]** | **Time** |
| tai27e | 41.50 | [0.11, 221.08] | 0.57 | **0.00** | [0.00, 0.00] | 0.08 |
| tai45e | 101.89 | [1.00, 400.60] | 3.83 | **0.00** | [0.00, 0.00] | 0.18 |
| tai75e | 111.28 | [6.20, 280.01] | 18.49 | **0.08** | [0.00, 0.63] | 1.27 |
| tai125e | 82.53 | [7.65, 265.54] | 72.52 | **3.65** | [0.78, 6.32] | 8.65 |
| tai175e | 67.86 | [9.11, 260.98] | 158.18 | **9.09** | [5.96, 12.24] | 14.43 |
| --- | --- | --- | --- | --- | --- | --- |
| **Average** | 81.01 | [4.82, 285.64] | 50.72 | **2.56** | [1.35, 3.84] | 4.92 |



## Contact
We hope that the package is useful for your application. If you have any questions related to the code or the paper, please feel free to email one of the following authors:
- Yicheng Pan, `panyicheng@stu.pku.edu.cn`
- Ruisong Zhou, `ruisongzhou@stu.pku.edu.cn`
- Haijun Zou, `haijunzou10853@gmail.com`
- Zaiwen Wen, `wenzw@pku.edu.cn`

## Reference
[Yicheng Pan, Ruisong Zhou, Haijun Zou, Tainyou Li & Zaiwen Wen.  "Learning to Solve the Quadratic Assignment Problem with Warm-Started MCMC Finetuning." arXiv preprint arXiv:2604.20109 (2026).](https://arxiv.org/abs/2604.20109)

## Citation
```bibtex
@article{pan2026learning,
      title={Learning to Solve the Quadratic Assignment Problem with Warm-Started MCMC Finetuning}, 
      author={Yicheng Pan and Ruisong Zhou and Haijun Zou and Tianyou Li and Zaiwen Wen},
      journal={arXiv preprint arXiv:2604.20109},
      year={2026},
}
```