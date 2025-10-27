# PLMA
PLMA is a permutation learning framework for the quadratic assignment problem (QAP). A QAP instance $\mathcal{P}$ is specified by two $n\times n$ matrices $F=(F_{ij})$ and $D=(D_{kl})$, where $F_{ij}$ is the flow  between facilities $i$ and $j$ and $D_{kl}$ is the  distance between locations $k$ and $l$. Let  $\Pi_n$ represent the set of all permutations over $\\{1,\dots,n\\}$. The QAP can be then formulated as 
$$\min_{\pi\in\Pi_n}\quad f(\pi;\mathcal{P}):=\sum_{i=1}^nF_{ij}D_{\pi(i),\pi(j)}.$$

## Algorithm
PLMA leverages a neural network to obtain parameterized probabilistic model $p_{\theta}(\pi\mid \mathcal{P})$ for each instance $\mathcal{P}$, through which the original optimization problem is transformed into a learning problem

$$\min_{\theta\in\mathbb{R}^d}\quad \mathcal{L}(\theta):= \mathbb{E}_{\mathcal{P}\sim \Gamma}\mathbb{E}_{\pi\sim p_{\theta}(\cdot\mid\mathcal{P})}[f(\mathcal{T}(\pi);\mathcal{P})],$$
where $\mathcal{T}$ is a local imporovement map used to smooth the underlying probability ditribution.

The learning process consists of two stages, where the model is first pre-trained on diverse instances to learn transferable structure prior and then fine-tuned on target instances for specialized efficacy. The finetuning procedure employs a unique warm-start mechanism inherit in MCMC sampling that reuses previous high-quality solutions to initialize customized short and locally-interacted Markov chains, thereby focusing the adaptation on promising regions. 

The probabilistic model utlized in PLMA is an energy-based model $\displaystyle p_{\theta}(\pi\mid\mathcal{P})=\frac{\exp(\Phi_{\theta}(\pi))}{Z_{\theta}}$ tailored for MCMC sampling. Specifically, the score function has an additive structure $\Phi_{\theta}(\pi) = \sum_{i=1}^n\phi_{i,\pi(i)}(\theta,\mathcal{P})$, where $\phi(\theta,\mathcal{P})=(\phi_{i,j}(\theta,\mathcal{P}))\in\mathbb{R}^{n\times n}$ is the heatmap output by the neural network. This structure enables $O(1)$-time evaluation of 2-swap proposals within a Metropolis-Hastings sampler. 

## Installation
The computational bottleneck of the PLMA framework, namely the parallelized execution of MCMC sampling and 2-swap local search, is addressed by high-performance implementations in CUDA C++. **Therefore, an NVIDIA GPU with a properly configured CUDA toolkit is a mandatory prerequisite for compilation and execution.**

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

## Evaluation
In our paper, we have evaluated PLMA on two families of synthetic QAP instances, and on the real-world QAPLIB and Taixxeyy benchmarks. Here we show how to run PLMA on these datasets and present partial results. 

### Dataset Preparation

### Running Scirpts

```
./scripts/run_sawt.sh
./scripts/run_uniform.sh
./scripts/run_qaplib.sh
./scripts/run_tai.sh
```
### Results
We compare our model against a wide spectrum of established and modern approaches, categorized as follows.
- **Search-based solvers:** A highly-optimized heuristic solver, Robust Tabu Search (Ro-TS);
- **Heuristic solvers:** Classic iterative algorithms commonly applied to graph matching, including IPFP, SM, and RRWM;
- **Learning-based solvers:** Two representative deep learning approaches, SAWT and NGM.

Results on two synthetic datasets are presented in the next two tables.  Some crucial conclusions can be drawn:
1. The pretrained model (PLMA, $T=1$) offers an instant, high-quality solution that already outperforms other learning-based approaches;
2. With brief fine-tuning (PLMA, $T=50$), the model rapidly converges to near-optimality, surpassing all other baselins than Ro-TS (5k);
3. With 200 fine-tuning steps, PLMA consistently matches or surpasses the strong Ro-TS (5k) baseline in quality while being remarkably more efficient. 
#### Geometrically Structured Dataset
|  |  | $n=50$| |  | $n=100$ | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algorithm**| **Cost** | **Gap** | **Time** | **Cost** | **Gap** | **Time** |
| Ro-TS (1k) | 375.99 | 0.14% | 4m35s | 1593.27 | 0.13% | 38m56s |
| Ro-TS (5k) | **375.48** | **0.00%** | 22m53s | 1591.25 | 0.00% | 3h15m |
| --- | --- | --- | --- | --- | --- | --- |
| IPFP | 378.76 | 0.88% | 11.47s | 1600.27 | 0.57% | 1m34s |
| IPFP (10) | 376.60 | 0.30% | 2m30s | 1594.76 | 0.22\% | 17m34s |
| RRWM | 428.78 | 14.14% | 39.23s | 1700.33 | 6.86% | 6m32s |
| SM | 426.92 | 13.70\% | 7.14s | 1753.10 | 10.17\% | 1m40s |
| --- | --- | --- | --- | --- | --- | --- |
| NGM | 429.69 | 14.46\% | 1m16s | 1773.71 | 11.47\% | 2m29s |
| SAWT (10k) | 380.92 | 1.45\% | 5m36s | 1617.30 | 1.64\% | 10m43s |
| --- | --- | --- | --- | --- | --- | --- |
| **PLMA ($\boldsymbol{T=1}$)** | 379.79 | 1.15\% | 0.41s | 1607.84 | 1.04\% | 4.27s |
| **PLMA ($\boldsymbol{T=50}$)** | 375.55 | 0.20\% | 19.88s | 1591.73 | 0.03\% | 3m30s |
| **PLMA ($\boldsymbol{T=200}$)** | **375.48** | **0.00\%** | 1m19s | **1591.23** | **0.00\%** | 13m58s |

#### Uniformly Random Dataset
|  |  | $n=50$| |  | $n=100$ | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Algorithm**| **Cost** | **Gap** | **Time** | **Cost** | **Gap** | **Time** |
| Ro-TS (1k) | 523.08 | 0.22\% | 4m36s | 2195.98 | 0.13\% | 38m59s |
| Ro-TS (5k) | 521.91 | 0.00\% | 22m59s | 2193.16 | 0.00\% | 3h15m |
| --- | --- | --- | --- | --- | --- | --- |
| IPFP | 530.74 | 1.69\% | 7.45s | 2211.38 | 0.83\% | 41.95s |
| IPFP (25) | 526.96 | 0.97\% | 4m20s | 2203.29 | 0.46\% | 19m20s |
| RRWM | 592.50 | 13.54\% | 31.91s | 2432.34 | 10.91\% | 5m6s |
| SM | 605.08 | 15.95\% | 5.65s | 2457.47 | 12.05\% | 1m23s |
| --- | --- | --- | --- | --- | --- | --- |
| NGM | 594.99 | 14.01\% | 1m17s | 2438.52 | 11.19\% | 2m29s |
| --- | --- | --- | --- | --- | --- | --- |
| **PLMA ($\boldsymbol{T=1}$)** | 538.42 | 3.17\% | 0.40s | 2243.43 | 2.29\% | 4.32s |
| **PLMA ($\boldsymbol{T=50}$)** | 523.95 | 0.40\% | 19.51s | 2200.40 | 0.34\% | 3m30s |
| **PLMA ($\boldsymbol{T=200}$)** | **521.83** | **-0.01\%** | 1m18s | **2193.13** | **0.00\%** | 14m1s |

#### QAPLIB Benchmark
The following table presents the average performance of different algorithms on the QAPLIB dataset, categorized by instance class. The metrics reported include the average optimality Gap and the computation Time (in seconds). It can be seen that PLMA achieves a near-zero average optimality gap on QAPLIB while being over 4 times faster than the strong Ro-TS baseline. 
|  | **Ro-TS** | | **SAWT** | | **IPFP** | | **PLMA** | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|**Class** | **Gap** | **Time** | **Gap** | **Time** | **Gap** | **Time** | **Gap** | **Time** |
| bur (26) | **0.00\%** | 0.12 | 3.95\% | 14.67 | 0.05\% | 0.33 | **0.00\%** | 0.07 |
| chr (12-25) | 0.48\% | 0.16 | 147.54\% | 14.18 | 14.90\% | 0.26 | **0.20\%** | 0.94 |
| els (19) | **0.00\%** | 0.02 | 47.37\% | 14.24 | 10.18\% | 0.36 | **0.00\%** | 0.08 |
| esc (16-128) | **0.00\%** | 0.62 | 43.29\% | 15.07 | 0.39\% | 0.63 | **0.00\%** | 0.04 |
| had (12-20) | **0.00\%** | 0.00 | 5.17\% | 14.23 | 0.08\% | 0.39 | **0.00\%** | 0.02 |
| kra (30-32) | **0.00\%** | 0.24 | 32.92\% | 14.77 | 0.65\% | 0.62 | **0.00\%** | 0.77 |
| lipa (20-90) | **0.03\%** | 3.28 | 1.40\% | 17.32 | 1.07\% | 4.93 | 0.10\% | 1.77 |
| nug (12-30) | **0.00\%** | 0.02 | 19.25\% | 14.39 | 0.05\% | 0.38 | **0.00\%** | 0.14 |
| rou (12-20) | **0.00\%** | 0.04 | 15.09\% | 14.25 | 0.77\% | 0.33 | 0.02\% | 1.77 |
| scr (12-20) | **0.00\%** | 0.01 | 33.92\% | 14.22 | 1.24\% | 0.28 | **0.00\%** | 0.04 |
| sko (42-100) | 0.05\% | 29.22 | 16.17\% | 19.06 | 0.30\% | 8.40 | **0.03\%** | 7.10 |
| ste (36) | **0.01\%** | 0.53 | 107.95\% | 14.95 | 1.81\% | 0.70 | **0.01\%** | 0.77 |
| tai (10-256) | **0.23\%** | 25.16 | 34.67\% | 16.62 | 0.92\% | 3.33 | 0.28\% | 4.93 |
| tho (30-150) | **0.04\%** | 38.74 | 24.05\% | 17.19 | 0.42\% | 12.46 | 0.11\% | 5.29 |
| wil (50-100) | **0.02\%** | 26.06 | 9.52\% | 17.64 | 0.12\% | 9.66 | **0.02\%** | 5.48 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Average** | 0.11\% | 9.68 | 37.82\% | 15.85 | 2.15\% | 2.73 | **0.10\%** | 2.29 |

#### Taixxeyy instances
The next table presents the average results of different algorithms on the Taixxeyy instance group, with all metrics being the mean values over 10 independent runs. For each class, the reported average and the [min, max] gaps are the averages of the per-instance statistics. The results show that PLMA delivers a low average gap of 2.38\% and a reliable worst-case performance with an average maximum gap of 3.69\%. This is in stark contrast to Ro-TS's highly erratic 81.01\% average gap and catastrophic failures with a maximum gap of 285.64\%. 
| | |**Ro-TS** | | | **PLMA**| |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class** | **[min, max]** | **mean** | **Time** | **[min, max]** | **mean** | **Time** |
| tai27e | [0.11\%, 221.08\%] | 41.50\% | 0.57 | [0.00\%, 0.00\%] | **0.00\%** | 0.08 |
| tai45e | [1.00\%, 400.60\%] | 101.89\% | 3.83 | [0.00\%, 0.35\%] | **0.03\%** | 0.28 |
| tai75e | [6.20\%, 280.01\%] | 111.28\% | 18.49 | [0.00\%, 0.45\%] | **0.09\%** | 1.48 |
| tai125e | [7.65\%, 265.54\%] | 82.53\% | 72.52 | [-0.08\%, 5.62\%] | **2.67\%** | 8.18 |
| tai175e | [9.11\%, 260.98\%] | 67.86\% | 158.18 | [5.61\%, 12.04\%] | **9.11\%** | 14.63 |
| --- | --- | --- | --- | --- | --- | --- |
| **Average** | [4.82\%, 285.64\%] | 81.01\% | 50.72 | [1.11\%, 3.69\%] | **2.38\%** | 4.93 |
