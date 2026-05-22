# BM sparseMarket Evaluation

This folder provides a self-contained BM sparseMarket evaluation module for PLMA. General PLMA background, CUDA requirements, and environment setup are described in the main [`README.md`](../README.md). This file only lists the BM-specific compilation, running commands, and results.

All commands below assume the current working directory is `bm/`.

## Setup

### Compilation of CUDA Kernels to Python Shared Library

The BM evaluation uses the CUDA backend in `src/backend/qap_solver_batch.cu`. The compiled shared library is included as:

```text
src/backend/qap_solver_batch.so
```

If the shared library is missing or incompatible with the local CUDA runtime, recompile it with:

```bash
cd src/backend
module load cuda
nvcc -Xcompiler -fPIC -shared -o qap_solver_batch.so qap_solver_batch.cu -lcublas -lcurand
cd ../..
```

The CUDA module name may vary across clusters. If the environment requires a target GPU architecture, add the corresponding flag, such as `-arch=sm_80`.

### Python Environment

The main PLMA environment is sufficient if it already contains the packages listed in `requirements.txt`. To create a standalone environment for this folder:

```bash
conda create -n qap-bm python=3.10
conda activate qap-bm
pip install -r requirements.txt
```

## Evaluation

### Running Scripts

```bash
./run_bm.sh
```

Alternatively, the evaluation can be launched directly after activating the environment:

```bash
python -u driver/eval_bm.py
```

The output CSV is written to:

```text
results/bm/results_bm_sparseMarket.csv
```

## Results

The following table reports the bandwidth minimization results on 62 BM sparseMarket instances. The metrics include the graph size $n$, the Reverse Cuthill-McKee bandwidth (RCM), the bandwidth obtained by PLMA, and the running time in seconds. PLMA improves the RCM bandwidth on 52 instances and matches it on the remaining 10 instances, reducing the average bandwidth from 43.06 to 27.65.

| Instance | $n$ | RCM | PLMA | Time (s) |
| :--- | ---: | ---: | ---: | ---: |
| GD06_theory.txt | 101 | 57 | 34 | 6.207 |
| GD96_c.txt | 65 | 16 | 9 | 0.534 |
| GD97_a.txt | 84 | 28 | 16 | 2.285 |
| GD98_c.txt | 112 | 52 | 25 | 9.225 |
| GD99_b.txt | 64 | 25 | 13 | 1.742 |
| Journals.txt | 124 | 117 | 98 | 5.658 |
| Sandi_authors.txt | 86 | 24 | 11 | 1.368 |
| Trefethen_150.txt | 150 | 79 | 67 | 7.797 |
| Trefethen_200.txt | 200 | 100 | 85 | 24.746 |
| Trefethen_200b.txt | 199 | 100 | 85 | 21.712 |
| Trefethen_300.txt | 300 | 152 | 128 | 71.068 |
| adjnoun.txt | 112 | 73 | 37 | 7.584 |
| ash292.txt | 292 | 34 | 21 | 32.683 |
| ash85.txt | 85 | 18 | 9 | 2.558 |
| bcspwr03.txt | 118 | 23 | 11 | 3.557 |
| bcspwr04.txt | 274 | 58 | 27 | 40.271 |
| bcsstk02.txt | 66 | 65 | 65 | 2.609 |
| bcsstk03.txt | 112 | 3 | 3 | 4.758 |
| bcsstk04.txt | 132 | 65 | 37 | 3.248 |
| bcsstk05.txt | 153 | 24 | 20 | 3.901 |
| bcsstk22.txt | 138 | 14 | 11 | 5.128 |
| bfwb62.txt | 62 | 11 | 8 | 0.410 |
| can_144.txt | 144 | 18 | 13 | 9.496 |
| can_161.txt | 161 | 30 | 18 | 8.268 |
| can_187.txt | 187 | 23 | 13 | 6.783 |
| can_229.txt | 229 | 48 | 30 | 19.371 |
| can_256.txt | 256 | 118 | 59 | 52.775 |
| can_268.txt | 268 | 133 | 52 | 31.689 |
| can_292.txt | 292 | 67 | 39 | 29.501 |
| can_61.txt | 61 | 26 | 13 | 1.057 |
| can_62.txt | 62 | 9 | 6 | 1.108 |
| can_73.txt | 73 | 28 | 16 | 1.521 |
| can_96.txt | 96 | 23 | 13 | 2.960 |
| dolphins.txt | 62 | 20 | 13 | 1.447 |
| dwt_162.txt | 162 | 21 | 13 | 7.057 |
| dwt_193.txt | 193 | 59 | 32 | 13.643 |
| dwt_198.txt | 198 | 14 | 8 | 9.285 |
| dwt_209.txt | 209 | 59 | 24 | 26.121 |
| dwt_221.txt | 221 | 16 | 14 | 9.256 |
| dwt_234.txt | 234 | 25 | 11 | 21.293 |
| dwt_245.txt | 245 | 44 | 23 | 19.257 |
| dwt_59.txt | 59 | 9 | 6 | 0.706 |
| dwt_66.txt | 66 | 3 | 3 | 0.472 |
| dwt_72.txt | 72 | 11 | 6 | 2.704 |
| dwt_87.txt | 87 | 17 | 10 | 3.332 |
| football.txt | 115 | 66 | 37 | 4.238 |
| grid1.txt | 252 | 19 | 19 | 18.894 |
| grid1_dual.txt | 224 | 17 | 17 | 16.543 |
| jazz.txt | 198 | 115 | 69 | 15.494 |
| lesmis.txt | 77 | 33 | 20 | 0.631 |
| lshp_265.txt | 265 | 18 | 17 | 15.773 |
| lund_a.txt | 147 | 23 | 23 | 3.576 |
| lund_b.txt | 147 | 23 | 23 | 3.590 |
| mesh3e1.txt | 289 | 17 | 17 | 9.867 |
| mesh3em5.txt | 289 | 17 | 17 | 10.267 |
| nos1.txt | 237 | 4 | 4 | 11.089 |
| nos4.txt | 100 | 15 | 10 | 1.933 |
| polbooks.txt | 105 | 39 | 20 | 1.829 |
| spaceStation_1.txt | 99 | 55 | 26 | 1.702 |
| sphere2.txt | 66 | 17 | 13 | 0.817 |
| sphere3.txt | 258 | 34 | 27 | 47.232 |
| tumorAntiAngiogenesis_1.txt | 205 | 199 | 100 | 25.120 |
| **Average** | **157.08** | **43.06** | **27.65** | **11.722** |
