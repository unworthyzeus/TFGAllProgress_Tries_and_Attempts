# Try 69 — Change Log

## Session 2026-04-14

### Problem diagnosed

After Try 69 started training, val RMSE was stuck at ~21 dB (worse than the analytic prior at 15 dB).
Validation breakdown showed:

| Regime | LoS RMSE | NLoS RMSE |
|---|---|---|
| open_lowrise (any ant.) | ~3–4 dB | **57–58 dB** |

The NLoS head was not learning — actively making predictions worse than the prior.

**Root cause:** `dual_los_nlos_head` with `out_channels=2`.
In open_lowrise environments (low building density), NLoS pixels are sparse.
The NLoS residual head received almost no gradient and produced random outputs
that degraded the prior prediction by 57+ dB. LoS head worked fine because
LoS pixels dominate in open/rural environments.

---

### Fix 1 — Disabled dual LoS/NLoS head

**File:** `scripts/generate_try69_configs.py`

```diff
- m["out_channels"] = 2
+ m["out_channels"] = 1   # single residual head for all pixels

- cfg["dual_los_nlos_head"] = {"enabled": True}
+ cfg["dual_los_nlos_head"] = {"enabled": False}
```

NLoS supervision is retained via `nlos_focus_loss` (auxiliary RMSE on NLoS pixels, weight per expert).
A single head receives gradient from all pixels (LoS + NLoS), which is enough to learn
NLoS corrections even in sparse-NLoS environments.

---

### Fix 2 — Added `--wipe-outputs` to submit scripts

**Files:** `cluster/submit_try69_experts_4gpu_sequential.py`,
`cluster/submit_try69_experts_2gpu_sequential.py`

New flag `--wipe-outputs` (implies `--no-resume`):
- Deletes `outputs/try69_expert_<id>/` on the cluster for each submitted expert
- Use when starting a fresh run that should not resume from any existing checkpoint

Example:
```bash
SSH_PASSWORD=... python cluster/submit_try69_experts_4gpu_sequential.py --cancel-all --wipe-outputs
```

---

### Run submitted after fixes (2026-04-14)

| Job | ID | Depends on |
|---|---|---|
| train open_lowrise | 10030518 | — |
| cleanup | 10030519 | 10030518 |
| train mixed_midrise | 10030520 | 10030519 |
| cleanup | 10030521 | 10030520 |
| train dense_highrise | 10030522 | 10030521 |
| cleanup | 10030523 | 10030522 |

Fresh start (all outputs wiped, no checkpoint resume).

---

## Session 2026-04-14 (follow-up) — training recipe bugs

### Bug A — CutMix incompatible with height-FiLM + height-dependent input channels

**Symptom:** Model sees contradictory physics: pasted spatial channels (formula prior, tx-depth, elevation, knife-edge) were computed in the dataset worker for **sample B’s** UAV height, while `scalar_cond` / FiLM still encode **sample A’s** height.

**Fix:** `train_partitioned_pathloss_expert.py` — skip spatial CutMix when `return_scalar_cond_from_config(cfg)` and `scalar_cond` is present. (CutMix can stay enabled in YAML; it simply no-ops under FiLM+scalar.)

### Bug B — Generator YAML disagreed with TRY69_DESIGN (LR + weight decay)

**Symptom:** `scripts/generate_try69_configs.py` set `learning_rate: 1e-3` and `weight_decay: 0.1` while TRY69_DESIGN and VERSIONS.md specified **3e-4** and **0.03**. The manual AdamW weight-decay path uses `param *= (1 - lr * wd)` each step; `wd=0.1` with `lr=1e-3` removes **1e-4 per step** from conv weights on top of optimizer effects — far stronger than the documented 3e-2 recipe.

**Fix:** Generator now emits `learning_rate: 3e-4`, `weight_decay: 0.03`. Patched the three checked-in expert YAMLs to match.

### Non-bug C — `corridor_weighting` (**pruned**)

The trainer never read `corridor_weighting`. **Change:** removed from `generate_try69_configs.py` and the three expert YAMLs so configs match code. Re-add only after implementing corridor weights in `train_partitioned_pathloss_expert.py`.

### Knife-edge vs formula carrier frequency (**aligned**)

YAMLs previously used **7.125** GHz for the hybrid formula but **3.5** GHz for knife-edge. **Change:** the generator copies `path_loss_formula_input.frequency_ghz` into `knife_edge_channel.frequency_ghz`; checked-in experts use **7.125** for both so the two analytic channels describe one carrier.

---

## Session 2026-04-14 (follow-up 2) — more code review notes

### D — Optimiser: PyTorch `weight_decay=0` + manual decay after `step`

`build_optimizer` constructs `AdamW(..., weight_decay=0.0)` and then `apply_optimizer_weight_decay` multiplies selected conv weights by `(1 - lr * wd)` once per optimiser step (after unscale/clip). There is **no second** decay inside AdamW. Tuning `training.weight_decay` in YAML refers to this manual path only.

### E — `generator_objective: full_map_rmse_only` skips regime reweighting on the loss mask

`train_partitioned_pathloss_expert.py` applies `_apply_regime_reweighting` only when `objective_mode != "full_map_rmse_only"`. Try 69 YAML keeps `regime_reweighting.enabled: false` anyway, so behaviour matches config today; if regime weights are enabled later, they will **not** affect the main RMSE branch in full-map mode unless the condition is revisited.

### F — DDP `find_unused_parameters=False`

`DistributedDataParallel(..., find_unused_parameters=False)` is the default in `main()`. Safe while every parameter receives loss signal; if optional heads or aux outputs are toggled off in config, a run could raise unused-parameter errors — switch to `True` only if needed (costs more).

### G — Learning-rate policy (documentation)

See **TRY69_DESIGN.md §2.1**: default remains **3e-4** per the anti-overfitting table; the thesis author documents a preference to **experiment upward** on LR once recipe bugs are fixed, rather than always lowering LR vs Try 66.

### H — Validation: prior RMSE used a different output policy than the model (**fixed**)

With `prior_residual_path_loss.clamp_final_output: true`, **`pred`** is passed through `clip_to_target_range` (per-expert `clip_min_db` / `clip_max_db`, e.g. open_lowrise 60–125 dB) before denormalizing for metrics. **`prior_path_loss`** used **`denormalize(prior)`** on the raw formula channel with **no** clamp — the analytic map can still follow the GT outside that band while the network cannot. That makes **`improvement_vs_prior`** and “worse than prior” statements **not apples-to-apples**.

**Fix:** `evaluate_validation` in `train_partitioned_pathloss_expert.py` — when `clamp_final`, apply `clip_to_target_range(prior, meta)` before `denormalize` for all prior-side RMSE accumulations (same policy as `pred`).

---

## Session 2026-04-14 (follow-up 3) — more reasons Try 69 can still underperform

These are **not new code bugs** after sessions A–H; they are recipe / formulation risks worth tracking when RMSE stays high.

1. **Fewer training samples per expert** than Try 66’s six-way split: each of the three morphology experts sees roughly half the city-filtered data per class vs the old quantile partition, so variance and overfitting pressure go up unless regularization compensates.

2. **`generator_objective: full_map_rmse_only`** builds `g_loss` from **main-map RMSE (or Huber) + multiscale + aux** (`nlos_focus`, `pde`, …). The tensors `final_loss` / `residual_loss` / `multiscale_loss` in the `else` branch are **zeroed** in this mode; there is **no separate MSE on `residual_pred` vs `residual_target`**. Gradients still flow through `pred = prior + residual`, but the optimisation narrative is “match GT in dB space,” not “explicit residual fit first.”

3. **Multiscale auxiliary uses the plain building mask** (`compute_multiscale_path_loss_loss(..., mask, ...)`) while the main RMSE can use **tail-focused or regime-weighted** masks in other objective modes. In `full_map_rmse_only` the main loss uses the unweighted mask too, so this is consistent today; if you switch objective mode later, multiscale may **under-weight** the same hard pixels the main loss emphasises.

4. **`pde_residual_loss`** penalises mean **absolute Laplacian** of **normalised** `pred` on **LoS ∩ valid** pixels. At weight `0.01` it is mild, but it still prefers smoother maps and can tug against sharp shadow boundaries (same tension as any smoothness prior).

5. **Routing vs ground truth:** experts train on `partition_filter.city_type`; holdout cities are still evaluated with the **same** router. If thresholds mis-classify morphology for an unseen city, the wrong expert’s clamp and data distribution can hurt.

6. **Strong regularisation stack:** dropout `0.2`, manual WD `0.03`, high `nlos_reweight_factor`, optional CutMix (now skipped when FiLM+scalar) — together they cap capacity; bad runs can be **under-fitting** rather than bugs.

`SOA_IMPLEMENTATION_STATUS.md` was updated so corridor weighting is not listed as implemented in the trainer.

---

## Session 2026-04-14 (follow-up 4) — tooling and config wiring

### Bug I — `evaluate.py` was unusable / wrong for Try 69 (**fixed**)

**Symptom:** The script imported `train_pmnet_residual` (not shipped under `TFGSixtyNinthTry69/`, so `python evaluate.py` from this folder failed) and always built **`PMNetResidualRegressor`**, ignoring `model.arch: pmhhnet` and FiLM. Metrics from a manual run could not match training validation.

**Fix:** `evaluate.py` now imports **`evaluate_validation`** and **`_build_pmnet_from_cfg`** from `train_partitioned_pathloss_expert.py`, builds **PMNet / PMHNet / PMHHNet** from config, and passes **`is_final_test=(split == "test")`** so TTA flags match training. Optional **`--use-ema`** loads `generator_ema` when present.

### Bug J — `save_validation_json_each_epoch` ignored (**fixed**)

**Symptom:** YAML flag was never read; JSON was always written.

**Fix:** `train_partitioned_pathloss_expert.py` calls `write_validation_json` only when `training.save_validation_json_each_epoch` is true (default **true** so existing runs unchanged).

### Note K — `predict.py` CLI defaults still point at Try 61

The **`--config` default** and argparse description mention Try 61; functionality supports PMHHNet. Prefer passing `--config` explicitly for Try 69 (cosmetic / footgun, not a training bug).

### Note L — Gradient accumulation + CutMix buffer

`prev_cutmix_buf` is updated **every** micro-batch inside the training loop. With `gradient_accumulation_steps > 1`, CutMix can still mix against the **previous micro-batch** from the **same** optimizer step, not only across steps. Unlikely to dominate RMSE vs the scalar-CutMix bug, but the augmentation statistics differ from “one mix per global batch.”

---

## Session 2026-04-15 — corridor modes, FiLM-safe CutMix, ablation driver

### 1. Corridor loss weights (Gao-style)

- **`corridor_weighting.mode: radial`** (default): cached Tx-centred Gaussian × inverse-square (pixels).
- **`anisotropic`**: cached elliptical tube (``angle_deg``, ``aspect_ratio``).
- **`gao_ray`**: each training step builds a **[B,1,H,W]** weight map from **mean building occupancy** along **K** samples on the ray map-centre → pixel (``num_ray_samples``, ``ray_obstruction_beta``), then multiplies the loss mask. Uses topology channel 0 × ``input_metadata.scale`` vs ``non_ground_threshold``.

### 2. CutMix compatible with FiLM (`cutmix_film_safe`)

When ``training.cutmix_film_safe.enabled: true`` and FiLM scalars are present, CutMix pastes **only** topology + LoS + distance from the buffered sample, then calls ``recompute_height_dependent_input_channels`` so formula / confidence / tx-depth / elevation / building / knife-edge match **current** ``scalar_cond`` (UAV height). If ``path_loss_obstruction_features.enabled`` is true, recompute is **skipped** (warn once) — turn obstruction off for this path or extend the recompute layout.

### 3. Short component ablations

- **`scripts/run_try69_component_ablation.py`**: same base YAML + seed; runs ``baseline``, ``no_corridor``, ``no_dual_head`` (``out_channels: 1`` + dual head off), ``no_pde``, ``no_multiscale`` with short epochs, disables early stopping, ``save_every: 1``, collects ``validate_metrics_latest.json`` → ``val_rmse_physical_db``.
- **`train_partitioned_pathloss_expert.py --epochs-override N`**: overrides ``training.epochs`` for smoke runs.

The three checked-in experts under ``experiments/sixtyninth_try69_experts/`` include ``cutmix_film_safe``, ``tx_oriented_cutmix``, and full ``corridor_weighting`` keys (defaults: ``mode: radial``). Re-run ``scripts/generate_try69_configs.py`` after changing the generator recipe so new experts stay in sync.
