# Next: subsample receivers + randomize for generalization

## Goal

Bring the **effective measurement density** closer to link-style benchmarks (few independent receivers per unit area) and avoid **fixed spatial grids** that let the model or metric exploit repeatable pixel locations. Two pieces:

1. **Cap receivers per unit area** — e.g. target roughly *N* valid ground receivers per **1000×1000 m²** of horizontal extent (or per fixed tile on the topology grid), instead of using almost every valid pixel as a supervised / scored point.
2. **Randomize which pixels survive** — each epoch (or each step), draw a **new** random subset of receivers subject to the cap, so the model cannot overfit a static subset of the map.

Together this moves evaluation and training pressure toward **sparse, spatially jittered** targets, closer in spirit to “one detector where you place it” than “every raster cell always counts.”

## Why

- **Pooled RMSE** today is correct but dominated by **mass and correlation**: huge \(N\) of pixels from the same scenes and similar geometry.
- **Fixed grids** encourage memorization of local context at fixed offsets from the Tx / image border.
- **Random sparse supervision** is a standard way to stress **generalization** and to approximate **variable receiver placement** without changing the simulator.

## Definition: “Rx per 1000”

Pick one convention and stick to it in code and thesis:

**Option A — geographic (preferred if you have `meters_per_pixel`):**

- Project valid ground pixels to world coordinates (or use a coarse **1000 m × 1000 m** binning of the scene extent).
- In each bin, keep at most **K** receivers per bin (e.g. \(K = 1\) or small integer).
- “Per 1000” then literally means **per km²**.

**Option B — image / grid (simpler, no geo):**

- Partition the **513×513** valid mask into non-overlapping **tiles** (e.g. 32×32 cells ≈ “per thousand pixels” if you choose tile size so expected valid count per tile is ~O(1000) — tune explicitly).
- Cap **K** scored pixels per tile.

Document the chosen **tile size**, **K**, and whether caps apply **per sample** or **globally per batch**.

## Randomization protocol

1. **Train:** after building `valid_mask`, apply `random_receiver_mask = subsample(valid_mask, cap=..., seed=epoch_or_step)` and use it in the **loss** (same mask for pred/target). Optionally use a **different** mask for auxiliary heads if needed.
2. **Val / test:** either  
   - **Single fixed seed** for reproducible reporting, or  
   - **Monte Carlo:** \(M\) random masks, report **mean ± std** of RMSE (more honest for stochastic scoring).

Recommendation: **train = random each step**; **val = fixed seed** for dashboards + optional MC block in `next_phase`.

## Metrics

- Keep **current pooled RMSE** on the **full** valid mask for continuity (rename in prose: “dense pixel RMSE”).
- Add **sparse RMSE** (same formula, only on subsampled pixels) and, if useful, **mean of per-map RMSE** under the sparse mask so large cities do not dominate by pixel count alone.

## Implementation sketch (repo)

- **`CKMHDF5Dataset` / batch path:** function `apply_receiver_subsample(mask, rng, cfg)` reading `data.receiver_subsample: { enabled, mode, tile_m, k_per_tile, seed_policy }`.
- **`train_partitioned_pathloss_expert.py`:** apply mask before `diff_phys` for loss and for metric totals (or separate metric pass on full mask once per epoch — heavier).
- **`evaluate_validation`:** optional flag `metrics.dense_and_sparse` to log both without doubling forward passes if mask is cheap.

## Risks / checks

- **Too sparse:** loss variance explodes; need minimum receivers per map or **warmup** with dense loss then fade in sparsity.
- **Leakage:** random must be **independent of target values**; only depend on indices / geometry / epoch seed.
- **Comparison to literature:** document equivalent **receiver density** (Rx per km²) next to RMSE so readers can relate to drive-test link density.

## Success criteria

- Sparse and dense RMSE **not** identical collapse (sanity: sparse slightly worse or comparable).
- **Val gap** vs train improves or NLoS tail improves when overfitting was partly grid-driven.
- Ablation table: dense vs sparse vs sparse+random seed sweep.
