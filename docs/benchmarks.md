# Benchmark interpretation

Full tables are in `benchmarks/results/summary.md` (regenerate them with `python3 benchmarks/run_benchmarks.py`). These results use **synthetic** price relatives and spending shares at realistic sizes. Re-running with real CPI series (`--cpi-series cu.data.0.Current`) is the next step. Machine: Apple M-series laptop, Python 3.12, NumPy 2.5; 7 timed runs after one warm-up; times are mean ± sd.

## 1. A1 vs A2 is a constant factor, not a different algorithm

Both do H·I·T multiply-adds. A1 scales with slope 1.01 in H, exactly as predicted. A2 is about 900× faster at every size (11.5 ms vs 0.013 ms at H = 2,000, I = 16, T = 12). The difference comes from interpreted Python versus compiled, vectorized BLAS, so the report labels it as a constant factor.

## 2. Measured slopes vs predicted complexity

| Experiment | Predicted (A2 / A4s) | Measured A2, largest 3 sizes | Measured A4s, largest 3 sizes |
|---|---|---|---|
| vary H (I = 250, T = 120) | 1 / 1 | 1.02 | 1.40 |
| vary I (H = 10k, T = 120) | 1 / 1 | 0.92 | 1.73 |
| vary T (H = 10k, I = 250) | 1 / 0 | 0.89 | −0.06 |
| vary G (H = 10k, I = 250, T = 120) | 0 / 0 | 0.00 | 0.15 |

- **A2 matches its O(H·I·T) prediction** at large sizes. At small sizes the slopes fall below 1 (0.6–0.8) because fixed call overhead dominates; the "largest 3" column removes most of that.
- **Aggregate first does not grow with T**, as predicted (slope ≈ 0). Once the G × I group shares exist, extra months cost only G·I·T, which is tiny.
- **A4s grows faster than linearly in H and I** at the top sizes. W is 10k × 250 doubles, about 20 MB, which spills out of cache, and SciPy's sparse kernel is single-threaded and memory-bound. Dense BLAS hides this better. The asymptotic model counts flops, not memory traffic.
- **The dense A4d really is O(G·H·I).** Its time grows with G (slope 0.90 at the top sizes), reaching 21 ms at G = 2,000, against 1.4 ms for sparse A4s. That is the problem the readers raised. With few groups (G = 5) dense A4d is the fastest version (2.6 ms vs 8.3 ms for A4s at H = 50k), because 5·H·I flops in BLAS beat a sparse kernel. **Use sparse when G is large and dense or grouped when G is small.**

## 3. The real workload: 45 weights × a sensitivity grid

To get standard errors, every group estimate is recomputed with the full-sample weight plus 44 replicate weights (K = 45), over S columns of a sensitivity grid. Stacking the 45 averaging matrices gives a sparse (45·G) × H matrix B with 45·H nonzeros. The two multiplication orders are:

- household first, B @ (W @ R): O(H·I·S + K·H·S)
- aggregate first, (B @ W) @ R: O(K·H·I + K·G·I·S)

Counting flops, aggregate first wins once S > K·I/(I + K): 11.8 columns for I = 16 and 38.1 for I = 250.

| I | Flop-count crossover | Measured, sparse B @ W | Measured, grouped BLAS |
|---|---|---|---|
| 16 | S ≈ 11.8 | S = 16 | S = 1 |
| 250 | S ≈ 38.1 | S = 256 | S = 16 |

The order matters, and so does sparse vs dense:

- With a **sparse** B @ W (single-threaded SciPy kernel), aggregate first costs a flat 23 ms at I = 250 and only beats household first at S = 256. Its flops are cheap on paper but slow in practice.
- The **grouped BLAS** version (`GroupedAggregator`) does the same K·H·I work as one dense K × H_g by H_g × I product per group. It costs a flat ~2 ms, so aggregate first wins from S = 16 at I = 250. At S = 256 it is 15× faster than household first (2.2 ms vs 30.7 ms).
- The flop-count crossover lands between the two measured ones. That is the expected pattern when two kernels with the same operation count run at different speeds per flop. A model of the form *time = flops ÷ kernel throughput* would predict both measured crossovers.

On the real CE weights (`python -m pie intrvw23`, section 5: 10,344 households, 5 quintiles, 45 weights) the stacked product reproduces the looped BRR standard errors to 2e-15. The grouped version takes about 0.15 ms whatever S is, against 1.2–9.2 ms for household first.

## Limits of these results

- Synthetic shares and prices. The timings depend on sizes, not values, but the real CPI item count and number of months should replace the synthetic I and T.
- One machine. BLAS threading and cache sizes change the constants, though not the exponents.
- At the smallest sizes timings are around 10 µs and noisy. Read the slopes from the larger sizes.
