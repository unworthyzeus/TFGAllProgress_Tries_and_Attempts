# Experiment Versions (Try 1 → Try 75)

This document summarizes the evolution from `TFG_FirstTry1` through the latest numbered try (`TFGSeventyFifthTry75` as of Try 75), what each family predicts, and why each new branch was opened.

## Current status

- **`Try 75`** (`TFGSeventyFifthTry75`) is the **second stage** of the new **3-expert no-prior curriculum**. It keeps the `Try 74` city-type experts (`open_lowrise`, `mixed_midrise`, `dense_highrise`), reopens the **full antenna-height range**, restores **scalar height conditioning + FiLM**, and by default **resumes from the matching `Try 74` checkpoint** for each expert. Configs: `scripts/generate_try75_configs.py` → `experiments/seventyfifth_try75_experts/try75_expert_*.yaml`. Plotters: `scripts/plot_try75_metrics.py`, `scripts/plot_try73_metrics.py`. Details: `TFGSeventyFifthTry75/README.md`.
- **`Try 74`** (`TFGSeventyFourthTry74`) is the **first stage** of that curriculum: **3 experts**, **no prior**, **tight antenna-height band `47.5–52.5 m`**, and **no explicit height modulation** (`scalar_feature_columns = []`, `use_scalar_channels = false`, `use_scalar_film = false`). It is designed to learn the morphology/path-loss mapping at near-constant geometry before reopening all heights. Configs: `scripts/generate_try74_configs.py` → `experiments/seventyfourth_try74_experts/try74_expert_*.yaml`. Plotters: `scripts/plot_try74_metrics.py`, `scripts/plot_try73_metrics.py`. Details: `TFGSeventyFourthTry74/README.md`.
- **`Try 73`** (`TFGSeventyThirdTry73`) is the **no-prior direct-prediction rerun** of the stable `Try 66/68` 6-expert PMHHNet family. It keeps the mature data/training layout, removes the formula prior and confidence channel, removes residual-over-prior optimization, keeps the building mask input, removes per-expert target clamps, and adds an **EMA non-finite guard** so invalid source tensors do not poison validation EMA weights. Configs: `scripts/generate_try73_configs.py` → `experiments/seventythird_try73_experts/try73_expert_*.yaml`. Plots: `scripts/plot_try73_metrics.py`. Details: `TFGSeventyThirdTry73/README.md`.
- **`Try 72`** (`TFGSeventySecondTry72`) is a **Try 68** fork with **sparse receiver supervision**: `data_utils.apply_receiver_subsample_mask` keeps roughly **`keep_fraction` of all originally valid ground pixels** (default **1%**, i.e. `0.01`), after a **per–1000 m tile** cap (`max_rx_per_tile`). Train uses a **new random mask each step** (seed from epoch/step); val uses **`val_seed`** for reproducibility. **Primary** `metrics.path_loss` RMSE is on the **sparse** mask; **`metrics.path_loss_dense_reference`** reports the same on **all valid pixels** for comparison to Try 68-style numbers. **Smaller PMHHNet** (`base_channels: 20`, `hf_channels: 10`) and **larger micro-batches** (`batch_size: 4`, `val_batch_size: 4`, `gradient_accumulation_steps: 4`). Configs: `scripts/generate_try72_configs.py` → `experiments/seventysecond_try72_experts/try72_expert_*.yaml`. Cluster: `cluster/submit_try72_experts_*`, `cluster/run_seventysecond_try72_*.slurm`. Plots: `scripts/plot_try72_metrics.py`. Details: `TFGSeventySecondTry72/README.md`.
- **`Try 71`** (`TFGSeventyFirstTry71`) is a **heteroscedastic uncertainty** fork of Try 68: the model predicts a **mean residual + log-variance** (2 output channels), trained with the Kendall & Gal (NeurIPS 2017) NLL loss `(μ-y)²/σ² + log σ²`. This automatically down-weights unresolvable NLoS pixels (high σ) and produces a **per-pixel confidence map**; evaluation reports RMSE at varying σ-threshold coverage levels. Only `open_sparse_lowrise`; resumes from Try 68 cluster checkpoint. Bootstrap: `TFGpractice/scripts/bootstrap_try71_from_try68.py`.
- **`Try 70`** (`TFGSeventiethTry70`) is an **experimental multi-scale quad-head** fork of Try 68: auxiliary residuals at 257 / 129 / 65 with quadrant tiling + global low-res branches; `train_try70_multiquad.py`. **Bug fixes applied (2025-04):** (1) global-branch double-counting in `try70_auxiliary_loss` (loops `range(5/17/65)` → `range(4/16/64)`); (2) aux heads supervising full target instead of residual target — fixed by passing `prior` to the loss; (3) blend search now accumulates SSE over the full val set (not just first batch) with alpha sweep [0.0…1.0].
- **`Try 69`** (`TFGSixtyNinthTry69`) — recipe from **Try 67** (dual LoS/NLoS heads, D4 TTA on val, SOA training stack) with **six `topology_class` experts** (Try 54 partition: `open_sparse_lowrise` … `dense_block_highrise`), **`knife_edge_channel` on**, **`path_loss_obstruction_features` off**; registry `experiments/sixtyninth_try69_experts/try69_expert_registry.yaml`. (Earlier revision used 3-class ITU YAMLs `open_lowrise` / `mixed_midrise` / `dense_highrise`.) **Bug fixes applied (2025-04):** (1) `lambda_recon: 0.0→1.0` and `mse_weight: 0.0→1.0` — the main 513px loss was completely disabled; (2) `generator_objective: full_map_rmse_only→legacy` — removes sqrt-loss instability at high error; (3) `corridor_weighting disabled` — sigma=40 on 513px map was centre-only, degrading edge RMSE; (4) `prior_residual_path_loss.loss_weight: 0→0.5` — no residual supervision; (5) regularisation relaxed (dropout 0.2→0.12, wd 0.03→0.015, cutmix 0.45→0.25); (6) clamp max +25 dB per expert. **SOA training added:** SWA (`start_fraction=0.6`), target label noise (`sigma_db=0.5`), LR score EMA smoothing (0.6). DataLoader: workers 0→6, batch 1→2, grad_accum 16→8.
- **`Try 68`** (`TFGSixtyEighthTry68`) continues the **Try 66** 6-expert PMHHNet synthesis + stem+HF fix + FiLM-safe CutMix. **Bug fixes applied (2025-04):** (1) `weight_decay: 0.1→0.015` — catastrophic regularisation caused pred≈prior, loss decreased while RMSE stalled; (2) `loss_type: huber→mse` inside `full_map_rmse_only` — switches to `masked_rmse_loss` (direct RMSE optimisation); (3) `corridor_weighting disabled` (same sigma=40 centre-only problem); (4) `prior_residual_path_loss.loss_weight: 0→0.5`; (5) `nlos_focus_loss enabled` (weight 0.2). **SOA training added:** SWA, target label noise, LR score EMA smoothing. Workers 4→6, batch 1→2, grad_accum 16→8.
- **`Try 67`** (`TFGSixtySeventhTry67`) remains the **3-class ITU** expert line with knife-edge channel + anti-overfitting recipe. **Bug fixes applied (2025-04):** same `lambda_recon/mse_weight/generator_objective/corridor/residual/regularisation/clamp` fixes as Try 69. **SOA training added:** SWA, target label noise, LR score EMA smoothing.
- **`Try 66`** (`TFGSixtySixthTry66`) synthesis try — unchanged (reference baseline at ~9.3 dB with overfitting plateau).
- `Try 20` and `Try 21` were useful controlled tests, but `Try 22` became the stronger clean `path_loss` baseline.
- `Try 22` established the best recent path-loss base:
  - bilinear decoder
  - multiscale path-loss supervision
- `Try 23` reopened the `delay_spread + angular_spread` branch with the same structural recipe adapted to continuous regression.
- `Try 24` was prepared locally as a multitask branch, but intentionally kept out of the active cluster reading so that the single-task results remain interpretable.
- `Try 25`, `Try 27`, and `Try 28` tested additional `path_loss` ideas around global context and topology-aware regularization.
- `Try 29` tested explicit radial supervision for `path_loss`, but did not beat the stronger `Try 22` baseline.
- `Try 30` tested stronger amplitude-aware spread supervision, but did not beat the stronger `Try 26` baseline.
- `Try 31` and `Try 32` were useful paradigm-shift tests, but they did not beat `Try 22` and `Try 26`.
- `Try 33` reuses the strong `Try 22` path-loss recipe and changes only one thing:
  - building pixels are excluded from supervision and error accumulation
- `Try 34` is the physical-prior path-loss branch:
  - hybrid two-ray + COST231 formula map as conditioning input
- `Try 35` is the lighter 1-GPU spread-side branch with the same building-mask exclusion
- `Try 36` is the clean `Try 26` spread baseline with the same building-mask exclusion
- `Try 37-40` restart that family on the newer `CKM_Dataset_270326.h5` dataset:
  - `Try 37`: `Try 33` equivalent on the new dataset
  - `Try 38`: `Try 34` equivalent on the new dataset, with a larger model than the tiny debug rerun
  - `Try 39`: `Try 35` equivalent on the new dataset
  - `Try 40`: `Try 36` equivalent on the new dataset
- `Try 41` opens the new main path-loss direction on the harder dataset:
  - `prediction = physical_prior + learned_residual`
  - with the hybrid two-ray/COST231 formula map acting as the prior anchor
- `Try 42` keeps the calibrated prior from `Try 41` but replaces the old U-Net/cGAN family with a PMNet-inspired residual regressor.
- `Try 43` is the PMNet control branch without any physical prior:
  - same masked training target
  - same LoS / distance / antenna context
  - but direct `path_loss` prediction without `prior + residual`
- `Try 44` is the more faithful PMNet-v3-style control branch:
  - still no physical prior
  - but with an encoder / ASPP / decoder structure much closer to the original PMNet repository
- `Try 43` and `Try 44` now also report the same regime-level diagnostics used in `Try 42`:
  - `LoS / NLoS`
  - `city type`
  - `antenna-height bin`
  - combined calibration-style regimes
- `Try 45` is the next step after `Try 42`:
  - same PMNet-style residual path-loss formulation
  - same `Try 22`-style good practices that still matter here:
    - bilinear resizing/fusion
    - multiscale path-loss loss
    - group norm for `batch_size = 1`
  - stronger train-only prior calibration using:
    - urban regime
    - `LoS / NLoS`
    - antenna-height bin
    - multiscale obstruction features
    - local shadow-support proxies from the LOS map
    - A2G-inspired `Eq. 9 / 10 / 11` strengthening for difficult `NLoS` cases
    - `Eq. 12`-style shadow-variability as a side feature
  - lightweight spatial mixture-of-experts residual head on top of the calibrated prior
  - `Try 45` stays off the cluster until prior-only validation reaches `NLoS RMSE < 20 dB`
- `Try 46` was the first explicit regime-split network for path loss:
  - shared PMNet-style trunk
  - small dedicated `LoS` residual head
  - stronger `NLoS`-only MoE residual head
  - branch-specific losses so `LoS` and `NLoS` do not collapse back into one average residual
- `Try 47` keeps the calibrated prior idea but returns to the stronger spatial base of `Try 22`:
  - bilinear U-Net decoder
  - group norm
  - scalar FiLM for antenna height
  - distance-map channel
  - explicit `prior + residual`
  - small `LoS` residual head
  - `NLoS`-only MoE residual head
  - explicit obstruction proxy channels:
    - shadow depth
    - distance since LoS break
    - maximum blocker height
    - blocker count
  - stronger combo losses for difficult `NLoS` subsets such as low-antenna deep-shadow cases
  - train-only CUDA calibration now runs as a separate logged job with sample-level subsampling so it can finish within the cluster time limit before training starts
- `Try 48` is a separated 2-stage pipeline:
  - Stage 1: PMNet base model for residual correction against the prior
  - Stage 2: frozen PMNet base + U-Net refiner trained with adversarial (GAN) detail pressure
  - inference: `prior + base_residual + refiner_residual`
- `Try 49` adds a prior confidence channel and widens the model to 112 channels:
  - prior confidence derived from LoS probability, distance map, and local obstruction density
  - gives the model spatial awareness of where the prior is reliable
  - uses `mae_dominant` loss weighting for the stage1 residual
  - a lightweight stage2 tail refiner trains on frozen stage1 teacher outputs
- `Try 50` is a prior-research sandbox focused on improving the `NLoS` prior formula:
  - explores height-dependent NLoS PLE (Vinogradov/Saboor family; PLE decays with altitude from ~4.5 to ~2.5)
  - explores A2G elevation-angle excess-loss for hard NLoS
  - best result still ~`41 dB NLoS RMSE`; branch archived as inconclusive
- `Try 51` is the literature-aligned reboot after `Try 49/50`:
  - supervised dense regression only (no GAN)
  - automatic city-type routing by density/height thresholds (not city-name lookup)
  - regime-aware loss reweighting during training (NLoS, low-antenna, dense-highrise)
  - city_holdout split enforced
  - transfer from an already-trained checkpoint instead of restarting from zero
- `Try 52` is a clean renamed version of the `Try 51` branch:
  - keeps the same paper-routed supervised recipe
  - adds stage3: a small global-context model refining only `NLoS` on top of stage2
  - cleaner config naming; stale `Try 51` paths removed
- `Try 53` is a cyclic feedback branch:
  - stages 1 → 2 → 3 → stage1 resume → 2 → 3 ...
  - metric-guided: stage1 regime weights are re-tuned from stage2/stage3 validation JSON before each stage1 resume
- `Try 54` introduces the **partitioned expert strategy and PMHHNet**:
  - 6 topology classes (`open_sparse_lowrise`, `open_sparse_vertical`, `mixed_compact_lowrise`, `mixed_compact_midrise`, `dense_block_midrise`, `dense_block_highrise`)
  - one PMHHNet specialist per class; a trained topology classifier routes at inference
  - PMHHNet = PMNet + lightweight high-frequency branch + continuous sinusoidal-FiLM height conditioning
  - calibrated physical prior from `Try 47` retained
- `Try 55` tests an aggressive objective change on the `Try 54` expert family:
  - generator optimized only with final-map RMSE + auxiliary no-data loss
  - no residual-only objective and no multiscale reconstruction term
  - aligns training loss directly with the reported metric
- `Try 56` is a topology-partitioned continuation of the old `Try 26` U-Net spread family for path loss:
  - same 6-expert partition as `Try 54`
  - adds binary `topology_mask` input channel
  - adds auxiliary `no_data` BCE head
  - stronger dropout than `Try 26`
- `Try 57` applies the `Try 55` objective change to the `Try 54` expert architecture on a fresh base
- `Try 58` and `Try 59` are topology-partitioned continuations of the `Try 26` U-Net family under alternative training settings
- `Try 60` is the **no-prior ablation of the 6-expert PMHHNet family**:
  - same 6 topology partitions and PMHHNet architecture
  - formula-prior channel and confidence channel removed
  - predicts `path_loss` directly from geometry + LoS context
  - measures whether the calibrated prior genuinely helps
- `Try 61` keeps the `Try 60` no-prior setup but attacks the hard regime directly:
  - strong LoS/NLoS regime reweighting in the generator loss
  - explicit NLoS loss term added
  - checkpoint selection based on composite proxy: `overall_rmse + alpha * nlos_rmse`
  - `open_sparse_vertical` split into two sub-experts: `_los` and `_nlos` (7 experts total)
- `Try 62` is the **paper-like reset after `Try 61`**:
  - back to 6 experts (removes the 7th vertical split)
  - restores the formula-prior input channel
  - adds obstruction proxy channels: `shadow_depth`, `distance_since_los_break`, `max_blocker_height`
- `Try 63` is a coarse-to-fine follow-up to `Try 62`:
  - stage1 trains at `128×128` (cheap and fast)
  - stage2 refines at full `513×513`
  - stage2 teacher upsamples stage1 prediction to full resolution before refinement
- `Try 64` is a coarse-to-fine follow-up to `Try 63` with adjusted hyperparameters
- `Try 65` is a **grokking-style stress test**:
  - single stage only, full `513×513`, no early stopping, no rewind to best model
  - very long horizon (10 000 epochs), high LR, high weight decay
  - tests whether long regularized training can achieve deep generalization without stage decomposition
- `Try 66` is the **synthesis try**: combines all validated ingredients with novel paper-backed additions:
  - 6-class topology-partitioned experts (validated from `Try 54`)
  - PMHHNet with sinusoidal height embedding → per-layer FiLM at 8 points (vs. 4 previously) — inspired by DDPM/ADM timestep conditioning
  - elevation angle map as an explicit input channel (Al-Hourani et al., 3GPP TR 38.901)
  - Tx-relative depth map (`building_height - antenna_height`) as input channel
  - propagation corridor weighting map in the loss (Gao et al. 2026)
  - single-stage direct 513×513 training (stage2 refiner dropped: never added >0.5 dB)
  - best result per expert: `open_sparse_lowrise` ~`9.3 dB` (plateaued/overfitting at epoch 96)
- `Try 67` replaces the arbitrary 6-class topology partition with a **physically motivated 3-class partition** (ITU-R P.1411 / 3GPP TR 38.901) and closes the train/val gap that caused `Try 66` to overfit:
  - 3 experts: `open_lowrise` (RMa/suburban), `mixed_midrise` (UMi), `dense_highrise` (UMa)
  - routing by building density/height thresholds — generalizes to unseen cities at inference without retraining
  - adds **knife-edge diffraction channel** (ITU-R P.526-15 §4.5.1 single-edge approximation)
  - anti-overfitting recipe: dropout ↑ 0.20, weight_decay ↑ 0.030, CutMix ↑ 0.45, `ReduceLROnPlateau` (replaces cosine warm restarts), MSE loss (replaces Huber)
  - **PDE residual loss** (masked Laplacian, ReVeal-style)
  - **D4 test-time augmentation** (8 orientations: identity, hflip, vflip, 180°, 90°, 270°, transpose, anti-transpose) at final test
  - per-expert tight output clamping (e.g., `open_lowrise`: 60–125 dB)
  - current active experiment

## Fast table

| Try | Folder | Main targets | Main idea |
|---|---|---|---|
| 1 | `TFG_FirstTry1` | `path_loss`, `delay_spread`, `angular_spread` | first working HDF5 multi-output pipeline |
| 2 | `TFGSecondTry2` | same three targets | stronger early multi-output baseline |
| 3 | `TFGThirdTry3` | same three targets | larger U-Net, more GPU-oriented |
| 4 | `TFGFourthTry4` | `path_loss` + confidence | first hybrid path-loss branch |
| 5 | `TFGFifthTry5` | same as 4 | more conservative optimization |
| 6 | `TFGSixthTry6` | same as 5 | switch to the larger HDF5 dataset |
| 7 | `TFGSeventhTry7` | `path_loss` | pure regression, no GAN |
| 8 | `TFGEighthTry8` | `path_loss` | reintroduce GAN |
| 9 | `TFGNinthTry9` | `path_loss` | antenna-height conditioning |
| 10 | `TFGTenthTry10` | `path_loss` | LoS / NLoS split |
| 11 | `TFGEleventhTry11` | `path_loss` | more formal height normalization |
| 12 | `TFGTwelfthTry12` | `path_loss` | early stopping + FiLM LoS/NLoS |
| 13 | `TFGThirteenthTry13` | `path_loss` | single FiLM model on the full dataset |
| 14 | `TFGFourteenthTry14` | `path_loss` | separate FiLM LoS and NLoS checkpoints |
| 15 | `TFGFifteenthTry15` | `path_loss` | first city-regime baseline |
| 16 | `TFGSixteenthTry16` | `path_loss` | conservative city-regime variant with more geometry |
| 17 | `TFGSeventeenthTry17` | `path_loss` | alternate city-regime balance |
| 18 | `TFGEighteenthTry18` | `path_loss` | additional city-regime regularization variant |
| 19 | `TFGNineteenthTry19` | `path_loss` | strongest city-regime base before opening new families |
| 20 | `TFGTwentiethTry20` | `path_loss` | bilinear decoder to reduce checkerboard artifacts |
| 21 | `TFGTwentyFirstTry21` | `path_loss` | multiscale path-loss loss |
| 22 | `TFGTwentySecondTry22` | `path_loss` | bilinear decoder + multiscale loss |
| 23 | `TFGTwentyThirdTry23` | `delay_spread`, `angular_spread` | bilinear decoder + multiscale regression loss |
| 24 | `TFGTwentyFourthTry24` | `path_loss`, `delay_spread`, `angular_spread` | local multitask branch, not launched |
| 25 | `TFGTwentyFifthTry25` | `path_loss` | `Try 22` base + lightweight bottleneck attention |
| 26 | `TFGTwentySixthTry26` | `delay_spread`, `angular_spread` | `Try 23` base + gradient-aware spread loss |
| 27 | `TFGTwentySeventhTry27` | `path_loss` | `Try 22` base + topology-edge path-loss weighting |
| 28 | `TFGTwentyEighthTry28` | `path_loss` | combine lightweight attention with topology-edge weighting |
| 29 | `TFGTwentyNinthTry29` | `path_loss` | `Try 22` base + radial profile loss + radial gradient loss |
| 30 | `TFGThirtiethTry30` | `delay_spread`, `angular_spread` | `Try 26` base + value-weighted spread loss + hotspot-focused spread loss |
| 31 | `TFGThirtyFirstTry31` | `path_loss` | physical prior + learned residual correction |
| 32 | `TFGThirtySecondTry32` | `delay_spread`, `angular_spread` | support map + amplitude map prediction |
| 33 | `TFGThirtyThirdTry33` | `path_loss` | `Try 22` recipe + building-mask exclusion only |
| 34 | `TFGThirtyFourthTry34` | `path_loss` | hybrid two-ray/COST231 formula input + building-mask exclusion |
| 35 | `TFGThirtyFifthTry35` | `delay_spread`, `angular_spread` | lighter 1-GPU spread-side branch with building-mask exclusion |
| 36 | `TFGThirtySixthTry36` | `delay_spread`, `angular_spread` | `Try 26` recipe + building-mask exclusion only |
| 37 | `TFGThirtySeventhTry37` | `path_loss` | `Try 33` rerun on `CKM_Dataset_270326.h5` |
| 38 | `TFGThirtyEighthTry38` | `path_loss` | `Try 34` rerun on `CKM_Dataset_270326.h5` with a larger model than the tiny debug variant |
| 39 | `TFGThirtyNinthTry39` | `delay_spread`, `angular_spread` | `Try 35` rerun on `CKM_Dataset_270326.h5` |
| 40 | `TFGFortiethTry40` | `delay_spread`, `angular_spread` | `Try 36` rerun on `CKM_Dataset_270326.h5` |
| 41 | `TFGFortyFirstTry41` | `path_loss` | `physical_prior + learned_residual` on `CKM_Dataset_270326.h5`, anchored to the hybrid formula map |
| 42 | `TFGFortySecondTry42` | `path_loss` | PMNet-inspired residual regressor over a calibrated physical prior |
| 43 | `TFGFortyThirdTry43` | `path_loss` | PMNet control branch without physical prior |
| 44 | `TFGFortyFourthTry44` | `path_loss` | more faithful PMNet-v3-style control branch without physical prior |
| 45 | `TFGFortyFifthTry45` | `path_loss` | `Try 42` + stronger train-only NLoS-aware prior + spatial MoE residual head, held until prior-only `NLoS < 20 dB` |
| 46 | `TFGFortySixthTry46` | `path_loss` | explicit `LoS / NLoS` branching over the calibrated prior, with `NLoS` experts only |
| 47 | `TFGFortySeventhTry47` | `path_loss` | return to `Try 22` U-Net + calibrated prior + explicit `LoS` head + `NLoS`-only MoE + obstruction proxy channels |
| 48 | `TFGFortyEighthTry48` | `path_loss` | 2-stage pipeline: PMNet stage1 + frozen base + U-Net GAN refiner (stage2); inference = prior + base_residual + refiner_residual |
| 49 | `TFGFortyNinthTry49` | `path_loss` | stage1 widened to 112 channels + prior confidence channel + MAE-dominant loss; stage2 tail refiner on frozen teacher |
| 50 | `TFGFiftiethTry50` | `path_loss` | prior-research sandbox: height-dependent NLoS PLE, A2G elevation-angle excess-loss; archived as inconclusive (~41 dB NLoS) |
| 51 | `TFGFiftyFirstTry51` | `path_loss` | literature-aligned reboot: supervised-only, automatic city-type routing, regime-aware loss reweighting, city_holdout split, transfer from trained checkpoint |
| 52 | `TFGFiftySecondTry52` | `path_loss` | clean `Try 51` continuation + stage3 NLoS global-context refiner; cleaner config naming |
| 53 | `TFGFiftyThirdTry53` | `path_loss` | cyclic feedback: stages 1→2→3→stage1 resume→...; metric-guided regime-weight updates between cycles |
| 54 | `TFGFiftyFourthTry54` | `path_loss` | **partitioned expert strategy + PMHHNet**: 6 topology experts, topology classifier routing, sinusoidal-FiLM height conditioning, HF branch |
| 55 | `TFGFiftyFifthTry55` | `path_loss` | `Try 54` experts with final-map RMSE-only objective (no residual-only term, no multiscale reconstruction) |
| 56 | `TFGFiftySixthTry56` | `path_loss` | `Try 26` U-Net family + 6-expert partition + topology_mask input + no_data BCE auxiliary |
| 57 | `TFGFiftySeventhTry57` | `path_loss` | `Try 55` objective on fresh `Try 54` expert base |
| 58 | `TFGFiftyEighthTry58` | `path_loss` | topology-partitioned `Try 26` U-Net family, alternative training settings |
| 59 | `TFGFiftyNinthTry59` | `path_loss` | topology-partitioned `Try 26` U-Net family, further alternative training settings |
| 60 | `TFGSixtiethTry60` | `path_loss` | no-prior ablation of the 6-expert PMHHNet family: direct path_loss prediction without prior channel |
| 61 | `TFGSixtyFirstTry61` | `path_loss` | `Try 60` + strong LoS/NLoS regime reweighting + composite checkpoint selection; vertical expert split into LoS/NLoS sub-experts (7 total) |
| 62 | `TFGSixtySecondTry62` | `path_loss` | paper-like reset: 6 experts, formula prior restored, obstruction proxy channels added |
| 63 | `TFGSixtyThirdTry63` | `path_loss` | coarse-to-fine: stage1 at 128×128, stage2 refiner at 513×513 |
| 64 | `TFGSixtyFourthTry64` | `path_loss` | `Try 63` coarse-to-fine with adjusted hyperparameters |
| 65 | `TFGSixtyFifthTry65` | `path_loss` | grokking-style stress test: single stage, 10 000 epochs, no early stopping, high LR + weight decay |
| 66 | `TFGSixtySixthTry66` | `path_loss` | **synthesis try**: sinusoidal FiLM at 8 points, elevation angle map, Tx depth map, corridor loss weighting, single-stage 513×513; best ~9.3 dB (overfitting plateau) |
| 67 | `TFGSixtySeventhTry67` | `path_loss` | **3-class ITU partition** + knife-edge diffraction channel + anti-overfitting recipe (dropout↑, CutMix↑, ReduceLROnPlateau) + PDE loss + D4 TTA |
| 68 | `TFGSixtyEighthTry68` | `path_loss` | **Try 66 synthesis rerun** on PMHHNet: stem + high-frequency injection fix, FiLM-safe CutMix guard, six topology experts; optional resume from Try 66 checkpoints |
| 69 | `TFGSixtyNinthTry69` | `path_loss` | **Try 67 recipe + SOA tooling** on **6 topology experts**, knife-edge **on**, obstruction **off**; dual LoS/NLoS head, D4 val TTA, FiLM-safe CutMix; `scripts/run_try69_component_ablation.py` |
| 70 | `TFGSeventiethTry70` | `path_loss` | **Try 68 + multi-scale quad heads**: same PMHHNet fused map → 513 residual plus aux at 257 (4 tiles + global), 129 (16+1), 65 (64+1); `train_try70_multiquad.py` + optional Try 68 init; blend RMSE report over full val set |
| 71 | `TFGSeventyFirstTry71` | `path_loss` | **Try 68 + heteroscedastic uncertainty**: model predicts mean residual + log-variance (2 channels), Kendall & Gal NLL loss, per-pixel confidence map, RMSE-vs-coverage evaluation; `open_sparse_lowrise` only, resumes from Try 68 checkpoint |
| 72 | `TFGSeventySecondTry72` | `path_loss` | **Try 68 + sparse receiver supervision**: train/val on reproducible sparse ground-pixel masks, dense-reference metric still reported; smaller PMHHNet, larger micro-batches |
| 73 | `TFGSeventyThirdTry73` | `path_loss` | **No-prior direct-prediction rerun** of the mature 6-expert PMHHNet line: formula/confidence channels off, residual-over-prior off, building mask kept, output clamps removed, EMA non-finite guard |
| 74 | `TFGSeventyFourthTry74` | `path_loss` | **3-expert no-prior fixed-height curriculum stage**: city-type experts, height band `47.5–52.5 m`, no scalar channels/FiLM, designed to learn near-constant-height morphology mapping |
| 75 | `TFGSeventyFifthTry75` | `path_loss` | **3-expert no-prior continuation from Try 74**: all heights reopened, scalar height FiLM restored, each expert resumes from the matching `Try 74` checkpoint |

## Main family transitions

## Tries 1-3: make the pipeline exist

The first three tries were about getting a working HDF5-based image-to-image pipeline for:

- `path_loss`
- `delay_spread`
- `angular_spread`

The main question at this stage was basic functionality and baseline stability, not yet strong physical specialization.

## Tries 4-14: path-loss specialization and physical conditioning

This phase progressively moved the project toward `path_loss` as the primary target.

Key additions across this stage:

- confidence-based hybrid path-loss prediction,
- larger and more stable HDF5 setup,
- pure-regression path-loss baselines,
- antenna-height conditioning,
- LoS / NLoS separation,
- FiLM-based scalar conditioning.

By `Try 14`, the project had a strong and stable path-loss-oriented base.

## Tries 15-19: city-regime consolidation

This family acted as a transition stage:

- cleaner dataset handling,
- conservative geometry-aware tuning,
- and a stronger baseline before opening more targeted architectural and loss-based branches.

The main value of this family was not radical novelty, but a cleaner reference point for future comparisons.

## Tries 20-22: establish the strongest modern path-loss base

- `Try 20` isolated the decoder hypothesis:
  - replace transposed convolutions with bilinear upsampling plus convolution.

- `Try 21` isolated the supervision hypothesis:
  - add multiscale path-loss loss.

- `Try 22` combined both.

This combination became the strongest clean recent path-loss branch and the base for later path-loss experiments.

## Tries 23 and 26: reopen and refine the spread branch

- `Try 23` transferred the structural improvements of `Try 22` to:
  - `delay_spread`
  - `angular_spread`

- `Try 26` added a gradient-aware spread loss because the outputs still looked too flat and blob-like.

These tries clarified that the spread branch needed not only a structural decoder/supervision update, but also better protection of local transitions.

## Tries 25, 27 and 28: extra path-loss hypotheses

- `Try 25` asked whether more global context was still missing, using lightweight bottleneck attention.
- `Try 27` asked whether path-loss errors should be weighted more strongly near topology edges and urban transitions.
- `Try 28` tested whether those two ideas were complementary when combined.

These branches were informative even when they did not clearly surpass `Try 22`, because they helped identify what the dominant remaining bottleneck was not.

## Try 29: radial path-loss supervision

After reviewing 20 composite diagnostic panels, the main path-loss conclusion was:

- the model still underlearns the transmitter-centered radial structure of the field.

That is why `Try 29` returns to the stronger `Try 22` base and adds:

- radial profile loss,
- radial gradient loss.

This is a more physically targeted follow-up than simply increasing model complexity again.

## Try 30: spread amplitude protection

The same visual review suggested that the spread branch often:

- gets the rough support approximately right,
- but underestimates sparse high-value responses.

That is why `Try 30` returns to the stronger `Try 26` spread base and adds:

- value-weighted spread regression loss,
- hotspot-focused spread loss.

This directly targets the amplitude imbalance visible in the `delay_spread` and `angular_spread` predictions.

## Try 31: physical prior + learned residual for path loss

`Try 31` is the first major formulation change for `path_loss`.

Instead of asking the model to predict the full map from scratch, it uses:

- a simple physical prior based on propagation distance and carrier frequency,
- plus a learned residual correction predicted by the network.

The idea is that the network should not have to rediscover the entire radial propagation law by itself. It should focus on learning how buildings, topology, and environment details deviate from that simpler baseline.

So far, the idea is conceptually strong but has not yet surpassed `Try 22`.

## Try 32: support + amplitude spread prediction

`Try 32` is the corresponding formulation change for:

- `delay_spread`
- `angular_spread`

Instead of directly predicting only the final value map, the network predicts:

- a support map telling where the response should exist,
- and an amplitude map telling how strong it should be there.

The final prediction is built from both. This is meant to help with the recurring spread failure mode where the approximate location is learned but the magnitude is underestimated.

So far, this idea is also promising in principle but has not yet surpassed `Try 26`.

## Tries 33-36: building-mask exclusion and new physical-prior branch

After reviewing the data interpretation more carefully, a simpler rule was introduced:

- pixels where `topology_map != 0` should not be treated as valid receiver locations;
- they should therefore be excluded from loss and error computation.

This produced two different experiment directions:

- `Try 33`:
  - keep the strong `Try 22` path-loss recipe,
  - but ignore building pixels during supervision and evaluation.

- `Try 34`:
  - open a separate path-loss branch with an explicit formula input,
  - using a hybrid two-ray / COST231 prior,
  - while also applying the same building-mask exclusion.

- `Try 35`:
  - keep a lighter 1-GPU spread-side branch active,
  - while still respecting the building-mask exclusion.

- `Try 36`:
  - return to the clean `Try 26` spread baseline,
  - apply the same building-mask exclusion,
  - and run it as the clearer 2-GPU comparison branch.

## Tries 37-40: same masked family, new dataset

After obtaining the newer dataset:

- `CKM_Dataset_270326.h5`

the masked-supervision family was restarted so that the dataset version and the masking policy could be tested together in a cleaner way.

- `Try 37`:
  - reruns the `Try 33` idea on the new dataset
  - `path_loss`
  - building-mask exclusion only

- `Try 38`:
  - reruns the `Try 34` idea on the new dataset
  - `path_loss`
  - hybrid `two_ray_ground + COST231` formula input
  - building-mask exclusion
  - larger than the tiny debug-sized `Try 34`

- `Try 39`:
  - reruns the lighter masked spread branch on the new dataset
  - `delay_spread`
  - `angular_spread`

- `Try 40`:
  - reruns the cleaner 2-GPU masked spread baseline on the new dataset
  - `delay_spread`
  - `angular_spread`

This restart also formalized a practical rule that was explicitly rechecked in code:

- building pixels are not only hidden in plots;
- they are removed from supervision;
- they are removed from metric accumulation;
- and they are exported as invalid (`NaN`) in error-map style visualizations.

## Try 41: make prior + residual the main path-loss formulation

The newer dataset appears harder than the old one, and the clean masked rerun (`Try 37`) dropped much more than expected.

That pushed the project toward a stronger formulation change:

- do not ask the model to predict the full `path_loss` map from scratch;
- instead ask it to predict a correction on top of a physically motivated prior.

`Try 41` therefore uses:

- `prediction = physical_prior + learned_residual`

with:

- the hybrid `two_ray_ground + COST231` formula map used as an explicit input channel,
- the same formula map also reused as the prior anchor for the residual target,
- building-mask exclusion still active.

The goal is to stop spending model capacity on rediscovering the basic radial propagation carrier and instead focus that capacity on the residual urban corrections.

An additional leakage-safe prior-only analysis was also added for this try:

- fit calibration on `train`,
- evaluate only on `val`,
- and score only on ground pixels where `topology == 0`.

That analysis showed:

- the raw prior is much too weak on its own (`~67.23 dB` RMSE on validation),
- but train-only regime-aware calibration improves it a lot,
- with the best prior-only variant so far reaching about `24.16 dB` on validation,
- using a quadratic calibration split by city type, LoS/NLoS, and antenna-height tertile.

This is still not enough to justify replacing the network with a hand-calibrated prior, but it does justify keeping the prior as a central scaffold for residual learning.

The rerun of `Try 41` therefore upgrades the prior from:

- raw hybrid formula input

to:

- train-only regime-aware quadratic calibration on top of that hybrid formula,
- split by city type, LoS/NLoS, and antenna-height tertile,
- still scored only on ground pixels where `topology == 0`.

The stored calibration and system description are documented in:

- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`

## Try 42: replace the U-Net/cGAN family with a PMNet-style residual regressor

`Try 42` is the first path-loss branch in this stage that changes the backbone family rather than just the prior usage or the loss balance.

It keeps the same calibrated physical prior used in `Try 41`, but changes the learning problem and the network:

- keep `prediction = calibrated_prior + learned_residual`;
- remove the discriminator path entirely;
- stop using the U-Net backbone;
- use a PMNet-inspired residual encoder plus dilated context module instead.

This decision is motivated by the local `TFG_Proto1` paper review, especially:

- `TFG_Proto1/docs/markdown/2402.00878v1 (2)/2402.00878v1 (2).md`

where PMNet is described as a stronger alternative than plain RadioUNet when longer-range propagation relationships matter.

`Try 42` also expands validation beyond the single global RMSE and now reports:

- global path-loss RMSE;
- global prior-only RMSE;
- RMSE by LoS and NLoS;
- RMSE by city type;
- RMSE by antenna-height bin;
- RMSE by the combined calibration regime.

So `Try 42` is not just “another try”.

It is the first branch in this family that tests whether a different path-loss backbone can exploit the calibrated prior better than the old U-Net/cGAN line.

The first received `Try 42` result showed an important pattern:

- the global RMSE remained poor (`~23.19 dB` at epoch 1),
- but the regime breakdown was highly asymmetric:
  - `LoS` already near `4.36 dB`
  - `NLoS` still around `40.46 dB`

This means the remaining difficulty is concentrated in urban correction regimes, not in the easy physical carrier itself.

## Try 43: PMNet control branch without prior

`Try 43` exists to answer a very simple control question:

- if the calibrated prior is removed entirely, how much of `Try 42` was due to the prior and how much was due to PMNet itself?

So `Try 43` keeps:

- PMNet-style path-loss regression,
- building-mask exclusion,
- LoS input,
- distance-map input,
- antenna-height conditioning.

But it removes:

- the physical prior input,
- the prior-residual decomposition,
- and the prior-only reporting block.

It predicts `path_loss` directly.

Even without the prior, it still reports the same regime-level metrics as `Try 42`, so failure can still be localized by:

- `LoS / NLoS`
- city type
- antenna-height bin
- combined calibration-style regimes

## Try 44: more faithful PMNet-v3-style control branch

`Try 44` is not just another PMNet control rerun.

It was opened because the first PMNet-inspired branch (`Try 42`) did not clearly outperform the prior-guided U-Net family, and it was still only PMNet-inspired rather than close to the official repository.

So `Try 44` keeps the same no-prior comparison setup as `Try 43`, but changes the backbone again:

- a more faithful PMNet-v3-style encoder,
- ASPP context aggregation closer to the original PMNet code,
- a decoder path closer to the original PMNet repository than the lightweight FPN-style fusion used in `Try 42`.

This makes `Try 44` the cleaner test of:

- whether PMNet itself is a better path-loss architecture for this project,
- independent of the physical prior.

## Practical rule for future tries

- If a change is mainly a data fix, execution fix, or cheap conditioning addition, it can be reused across a family.
- If a change alters the decoder, the loss, the target definition, the routing, or the main architecture, it should become a new try.

## Try 45: enhanced prior + spatial MoE residual head

`Try 45` is the gated follow-up to `Try 42`.

The `Try 42` result showed a clear pattern:
- `LoS` was already relatively controlled by the prior (~3.86 dB).
- `NLoS` remained the dominant failure mode (~34.47 dB).

`Try 45` therefore attacks both sides:

**Prior side** — adds to the prior:
- train-only calibration by regime;
- multiscale local topology descriptors;
- local shadow-support proxies from the LoS map;
- A2G-inspired elevation terms for hard `NLoS`.

**Residual side** — spatial MoE head:
- shared PMNet-style encoder/context backbone;
- several small residual experts;
- per-pixel gating maps;
- expert-balance regularization.

Launch policy: `Try 45` is gated and should only reach the cluster when prior-only validation reaches `NLoS RMSE < 20 dB`.

## Try 46: explicit LoS/NLoS branching

`Try 46` is the first branch that treats `LoS` and `NLoS` as structurally different prediction problems inside the network.

The model keeps:
- calibrated physical prior;
- PMNet-style shared trunk;
- `prior + residual` formulation.

But replaces the single residual head with:
1. a lightweight `LoS` residual head supervised only on `LoS` valid pixels;
2. a stronger `NLoS`-only MoE residual head supervised only on `NLoS` valid pixels.

The final residual is blended by the explicit `LoS` map. Branch-specific losses prevent both heads collapsing to the same average solution.

`Try 46` also expanded the validation JSON to report `NLoS` by shadow depth: `shallow_shadow`, `medium_shadow`, `deep_shadow`.

## Try 47: Try-22 U-Net + calibrated prior + LoS/NLoS specialization

`Try 47` is a deliberate synthesis of three prior branches:
- `Try 42` contributed the `prior + residual` idea;
- `Try 46` contributed the `LoS / NLoS` regime split;
- `Try 22` contributed the stronger image-to-image spatial backbone.

Architectural recipe:
- bilinear upsampling decoder (no transposed-convolution checkerboard);
- GroupNorm for `batch_size = 1`;
- scalar FiLM conditioning for antenna height;
- explicit distance-map input channel;
- multiscale path-loss supervision;
- one lightweight `LoS` residual head;
- one `NLoS`-only MoE residual head;
- four obstruction proxy channels: `shadow_depth`, `distance_since_los_break`, `max_blocker_height`, `blocker_count`.

Extra weighted combo losses target the hardest `NLoS` subsets:
- `low_ant + deep_shadow`;
- `mid_ant + deep_shadow`;
- `dense_highrise` subsets.

The prior calibration now runs as a separate cluster job with an `afterok` dependency so the network only starts after the calibrated prior JSON has been refreshed.

## Try 48: 2-stage separated pipeline (PMNet + GAN refiner)

`Try 48` separates refinement into two explicit stages:
- Stage 1: PMNet base model trained with `optimize_residual_only: true`, `lambda_gan: 0`.
- Stage 2: frozen Stage 1 base + a new U-Net refiner trained with adversarial pressure (`lambda_gan > 0`).
- Inference: `prior + stage1_residual + stage2_refiner_residual`.

Stage 2 only starts after Stage 1 exits (SLURM `afterany` dependency). Stages write to separate output folders to avoid checkpoint mixing.

## Tries 49–53: prior refinement and literature alignment

`Try 49` adds a **prior confidence channel** heuristically derived from `LoS` probability, distance, and local obstruction density, giving the model spatial awareness of prior reliability. The model is widened to 112 channels and uses a MAE-dominant stage1 loss. A lightweight stage2 tail refiner trains on frozen stage1 teacher outputs.

`Try 50` is a **prior-research sandbox** testing height-dependent NLoS PLE (Vinogradov/Saboor family) and A2G elevation-angle excess-loss. Best NLoS result remained ~41 dB RMSE; all variants archived as `worse_experiments/`. Practical conclusion: hand-tuning the formula alone cannot fix hard NLoS.

`Try 51` is the **literature-aligned reboot**. Key changes from `Try 49`:
- supervised dense regression only (no GAN);
- automatic city-type routing by density/height thresholds (no city-name lookup);
- regime-aware loss reweighting during training;
- city_holdout split strictly enforced;
- transfer from an already-trained checkpoint.

`Try 52` is a clean rename of the `Try 51` line with an added stage3 NLoS global-context refiner.

`Try 53` adds a **cyclic feedback loop**: stage1 is resumed after stage2/stage3 are trained, with stage1 regime weights re-tuned from stage2/stage3 validation metrics. The second cycle then retrains stage2 and stage3 on the updated stage1 teacher.

## Try 54: partitioned experts + PMHHNet

`Try 54` is the first branch that fully embraces a **partitioned expert strategy** instead of a single universal regressor.

Four linked ideas:
1. **6 topology experts**: `open_sparse_lowrise`, `open_sparse_vertical`, `mixed_compact_lowrise`, `mixed_compact_midrise`, `dense_block_midrise`, `dense_block_highrise`.
2. **Topology classifier**: a small network predicts the topology class from the input map; routing uses topology alone, not antenna height.
3. **PMHHNet**: PMNet + lightweight high-frequency branch + continuous sinusoidal-FiLM height conditioning inside the feature hierarchy.
4. **Height generalization**: each expert handles all antenna heights via the FiLM mechanism, not by splitting into height-specific sub-models.

The `Try 47` calibrated prior is retained. Per-expert RMSE is consistently lower than monolithic models.

## Tries 55–59: objective and structural variants on the 6-expert family

`Try 55` changes the generator objective on the `Try 54` experts to **final-map RMSE only** (plus no-data auxiliary), removing the residual-only term and multiscale reconstruction. Aligns training loss directly with the reported metric.

`Try 56`, `Try 58`, `Try 59` are topology-partitioned extensions of the older `Try 26` U-Net spread family, testing the same partition idea on a simpler backbone with building-mask and no-data auxiliary head.

`Try 57` applies the `Try 55` objective change to a fresh `Try 54` base.

## Try 60: no-prior ablation

`Try 60` removes the formula-prior channel from the 6-expert PMHHNet family to measure whether the calibrated prior was genuinely helping or anchoring the model into a bad low-frequency bias. Predicts `path_loss` directly from geometry and LoS context.

## Try 61: NLoS-focused objective + vertical expert split

`Try 61` keeps the `Try 60` no-prior setup but adds:
- strong LoS/NLoS regime reweighting during training;
- explicit NLoS loss term in the generator;
- composite checkpoint selection: `overall_rmse + alpha * nlos_rmse`.

The `open_sparse_vertical` expert is split into `_los` and `_nlos` sub-experts (7 total).

## Try 62: paper-like reset

`Try 62` moves back toward the strongest patterns in the literature:
- 6 experts (removes the 7th vertical split);
- formula-prior input channel restored;
- obstruction proxy channels added: `shadow_depth`, `distance_since_los_break`, `max_blocker_height`.

## Tries 63–64: coarse-to-fine pipeline

`Try 63` and `Try 64` test a coarse-to-fine pipeline:
- stage1 trains at 128×128 (cheap, fast, learns global structure);
- stage2 refines at full 513×513 (teacher upsamples stage1 to full resolution before refinement).

Result: stage2 refiner added <0.5 dB gain — below the 1 dB quantization floor of the uint8 ground truth. Removed in `Try 66`.

## Try 65: grokking-style stress test

`Try 65` tests whether a stubborn long-horizon training regime can achieve deep generalization without stage decomposition:
- single stage, full 513×513, 10 000 epoch horizon;
- no early stopping, no rewind to `best_model`;
- large constant LR, high weight decay.

## Try 66: the synthesis try

`Try 66` is the deliberate synthesis of everything validated across 65 prior tries with novel paper-backed additions.

**Core innovation: continuous multi-altitude prediction.** The central contribution is a single model that predicts path loss at arbitrary continuous antenna heights by deeply conditioning the entire network on the height scalar. No existing radio-map neural network handles height this deeply.

**How height enters the model (7 levels):**

| Level | Mechanism | Source |
|---|---|---|
| 1 | Physical prior channel (two-ray/COST231) uses `h_tx` directly | — |
| 2 | Tx-depth map: `building_height − h_tx` per pixel | Gao et al. 2026 |
| 3 | Elevation angle map: `atan2(h_tx − h_rx, d)` per pixel | Al-Hourani et al. 2014; 3GPP TR 38.901 |
| 4 | Sinusoidal embedding: 64-dim (32 sin/cos pairs), resolves ~0.3 m | DDPM (Ho et al. 2020) |
| 5 | Per-layer FiLM at 8 points in the network | ADM (Dhariwal & Nichol 2021) |
| 6 | SE channel attention operating on FiLM'd features | — |
| 7 | ~~Stage 2 height FiLM~~ — removed; single-stage 513×513 | — |

**What is kept (validated across multiple tries):**
- 6-class topology-partitioned experts (Try 54)
- PMHHNet backbone (Try 54)
- Calibrated physical prior + residual learning (Try 41–42)
- Building mask exclusion (Try 33)
- City-holdout data split (Try 51)
- EMA decay 0.99 (Try 55)
- GroupNorm (Try 22)
- Geometric augmentation hflip/vflip/rot90 (Try 22+)
- Multiscale path-loss loss (Try 22)
- Supervised regression only — no GAN (Try 51)

**What is new (paper-backed):**
- Sinusoidal height embedding + per-layer FiLM at 8 points (DDPM/ADM pattern)
- Elevation angle map as input channel (Al-Hourani/3GPP)
- Tx-relative depth map as input channel (Gao et al. 2026)
- Propagation corridor weighting map in the loss (Gao et al. 2026; ~0.55–1.16 dB gain reported)
- Single-stage direct 513×513 (stage2 dropped: <0.5 dB and unjustified complexity)

**Reference numbers from cluster:**

| Try | Setting | Overall RMSE | LoS RMSE | NLoS RMSE |
|---|---|---|---|---|
| 42 | PMNet + prior (single) | ~19.78 dB | ~3.86 dB | ~34.47 dB |
| 49 | PMNet stage1 w112 | ~18.96 dB | — | — |
| 55 | PMHHNet expert (open_sparse_lowrise) | ~10.53 dB | ~3.76 dB | ~35.82 dB |
| 64 | Coarse 128 + refiner (open_sparse_lowrise) | ~7.76 dB (128px) | ~2.97 dB | ~24.91 dB |
| 66 | Synthesis try (open_sparse_lowrise) | ~9.3 dB | — | — |

**Problem:** `Try 66` plateaued and started overfitting after the best epoch (e.g., `open_sparse_lowrise` stalled at 9.34 dB). Root cause: cosine warm-restarts reset LR to peak at exactly the epochs when overfitting began; Huber loss with δ=6 dB masked NLoS tail errors; dropout and weight decay too weak for 400-sample expert datasets.

**Architecture note (discovered later):** In `PMHHNetResidualRegressor.forward`, when FiLM height conditioning was added on top of `PMHNetResidualRegressor`, the **stem no longer added** the high-frequency map `hf_project(|∇²x|)` to the stem output. `PMHNet` does `x0 = stem(x) + hf` so building edges drive the whole encoder; the buggy `PMHHNet` path only concatenated `hf` at the FPN fusion, starving early layers of that signal. The fix restores `stem(x) + hf` before `film_stem` and reuses the same `hf` tensor for `film_hf` + fusion. See **Try 68** and `TFGSixtySeventhTry67/model_pmhhnet.py`.

## Try 67: 3-class ITU partition + anti-overfitting + knife-edge diffraction

`Try 67` addresses the two problems from `Try 66`: arbitrary topology partition and overfitting.

**Partition change: 6 classes → 3 city-morphology classes (ITU-R P.1411 / 3GPP TR 38.901):**

| Expert | Morphology | ITU/3GPP analogue |
|---|---|---|
| `open_lowrise` | low density, low height | RMa / suburban |
| `mixed_midrise` | medium density, medium height | UMi street-canyon |
| `dense_highrise` | high density, tall buildings | UMa |

Routing uses fixed thresholds on building density + mean building height from a calibration JSON — no city-name lookup, generalizes to unseen cities at inference.

**New input channel: knife-edge diffraction (ITU-R P.526-15 §4.5.1):**
- For every pixel, a ray is cast from TX (map center, altitude `h_tx`) to the receiver pixel (`h_rx = 1.5 m`).
- 48 samples along the ray; dominant blocker found via argmax of excess height above the straight TX→RX line (Bullington 1947 single-edge approximation).
- Fresnel parameter `v = h · √(2(d₁+d₂)/(λ·d₁·d₂))`; Lee 1985 closed form for `J(v)`.
- Map normalized to [0,1] by a 40 dB scale; appended as extra input channel.
- Expected gain: 1–3 dB NLoS.

**Anti-overfitting recipe (all changes motivated by `Try 66` training logs):**

| Knob | Try 66 | Try 67 | Reason |
|---|---|---|---|
| `dropout` | 0.12 | 0.20 | Gap opened at epoch 97 with 0.12 |
| `weight_decay` | 0.015 | 0.030 | Too low for 400-sample experts |
| `cutmix_prob` | 0.25 | 0.45 | Zhang et al. MICCAI 2024: strongest regularizer for limited dense tasks |
| `learning_rate` | 4e-4 | 3e-4 | Lower peak LR prevents instant re-overfitting after rewind |
| `lr_scheduler` | cosine_warm_restarts | reduce_on_plateau | Warm-restart spikes reset LR to peak right when overfitting began |
| `loss_type` | huber (δ=6) | mse | Huber's tolerance masked NLoS tail errors |
| `epochs` | 1000 | 800 | Early stopping trips much earlier anyway |
| `early_stopping.patience` | 50 | 25 | Stop wasting ~25 epochs after the best epoch |
| `ema_decay` | 0.995 | 0.9975 | Slower EMA averages more weights, smoother validation |
| `lr_warmup_steps` | 0 | 500 | Gentle warmup from 10% LR avoids large early gradients |

**Additional techniques:**
- **PDE residual loss** (masked Laplacian, ReVeal-style, arXiv:2502.19646): expected 0.3–1.0 dB NLoS gain.
- **D4 test-time augmentation** at final test: 8 orientations (identity, hflip, vflip, 180°, 90°, 270°, transpose, anti-transpose); expected 0.3–1.0 dB.
- **Per-expert tight output clamping**: `open_lowrise` [60, 125] dB; `mixed_midrise` [58, 135] dB; `dense_highrise` [55, 145] dB.
- **LR warmup**: 500 steps from 10% of base LR.

**PMHHNet stem+HF fix (same codebase family):** `TFGSixtySeventhTry67/model_pmhhnet.py` applies the `stem(x) + hf` correction described under Try 66 above so new Try 67 runs use the intended PMHNet signal path.

### Bug fixes applied to Try 67 (2025-04)

Six configuration bugs were identified that collectively disabled most of the training signal:

| Bug | Bad value | Fixed value | Impact |
|-----|-----------|-------------|--------|
| `loss.lambda_recon` | 0.0 | 1.0 | Main 513px reconstruction loss was completely disabled |
| `loss.mse_weight` | 0.0 | 1.0 | MSE term inside reconstruction also zero |
| `training.generator_objective` | `full_map_rmse_only` | `legacy` | Switched from `sqrt(MSE)` (unstable gradient at high error) to proper MSE |
| `corridor_weighting.enabled` | true (sigma=40) | false | sigma=40 on 513px map → ~0 weight at edges; degraded far-field RMSE |
| `prior_residual_path_loss.loss_weight` | 0.0 | 0.5 | No residual supervision at 513px |
| `regularisation` | dropout=0.2, wd=0.03, cutmix=0.45 | 0.12 / 0.015 / 0.25 | Over-regularised for 3-class experts with ≥400 training samples |
| `clip_max_db` | `open_lowrise` 125 dB | 150 dB | Clipped valid high-path-loss predictions in open expert |

### SOA training additions to Try 67 (2025-04)

Three SOA training techniques added to `train_partitioned_pathloss_expert.py` (shared across Try 67 / 68 / 69):

**1. Stochastic Weight Averaging (SWA)** — Izmailov et al. NeurIPS 2018: running uniform average of checkpoints from `swa.start_fraction × total_epochs`. Finds wider loss-surface optima; reported +0.5–1.0 dB on city-holdout generalization vs standard EMA. Implemented as `_update_swa_model()` helper; activated by `training.swa.enabled: true` + `swa_lr` in YAML.

**2. Target label noise** — Müller et al. 2019 (label smoothing generalization): Gaussian noise `N(0, 0.5 dB)` added to uint8 training targets. The CKM dataset stores path-loss at 1 dB integer resolution; the model tends to predict constant-integer-level plateaus and the loss surface has grid-aligned narrow valleys. Adding σ=0.5 dB noise (≈0.5/180 normalized) breaks the quantization grid while staying below the dataset's physical uncertainty floor. Activated by `training.target_noise.enabled: true` + `sigma_db` in YAML.

**3. LR-score EMA smoothing** — `training.lr_scheduler_score_ema: 0.6` smooths the RMSE fed to `ReduceLROnPlateau`. Raw per-epoch RMSE has ~0.3 dB noise from batch sampling; without smoothing the plateau scheduler triggers 3–5 epochs early, dropping LR while the model is still improving. The smoothed score is stored on the scheduler object (`_smoothed_score`) and passed to `scheduler_g.step()`.

## Try 68: Try 66 recipe + PMHHNet stem+HF bugfix + cluster resume from Try 66

`Try 68` is not a new research direction: it **reuses the full Try 66 synthesis** (6 topology experts, same losses, configs, and cluster layout) with the **PMHHNet forward fix** (high-frequency branch injected at the stem, aligned with `PMHNetResidualRegressor`).

**Training fix (Try 68 only vs Try 66 fork):** In `train_partitioned_pathloss_expert.py`, **CutMix is disabled whenever** `return_scalar_cond_from_config(cfg)` **and** `scalar_cond` **are set**. Try 66 still applies CutMix on `x`/`y`/`m` without mixing the height vector, which misaligns height-built input channels with FiLM; Try 68 matches the guard used in Try 67 for the same reason.

**Folder:** `TFGpractice/TFGSixtyEighthTry68/` (generated from `TFGSixtySixthTry66` via `TFGpractice/scripts/bootstrap_try68_from_try66.py`).

**Naming on disk and cluster:** `try68_expert_*`, `sixtyeighth_try68_experts/`, Slurm scripts `run_sixtyeighth_try68_*.slurm`, submitters `submit_try68_*`. Default `MASTER_PORT` / `base-master-port` offsets differ from Try 66/67 to avoid collisions if multiple tries run concurrently.

**Checkpoints:** Same parameter tensors as Try 66 (no new layers); training can **resume** from Try 66 `best_model.pt` files after copying them into Try 68 output directories on the cluster.

**Cluster helpers:**

| Script | Role |
|--------|------|
| `cluster/launch_try68_resume_from_try66.py` | SSH: `cp -a` each `outputs/try66_expert_*` → `outputs/try68_expert_*` on the cluster (same `expert_id` suffix). Optional `--submit` runs upload with `--no-clean-outputs`, sync, then `submit_try68_experts_4gpu_sequential.py --skip-upload`. |
| `cluster/submit_try68_experts_4gpu_sequential.py` | Chained **6 train (4× RTX2080) + 6 cleanup** jobs; pass **`--no-clean-outputs`** with upload if `outputs/` was populated by the sync first (otherwise the uploader’s default remote output clean would delete resumed checkpoints). |

### Root-cause analysis and bug fixes applied to Try 68 (2025-04)

**Root cause of loss↓ / RMSE stall** (confirmed from cluster epoch-164 JSON for `open_sparse_lowrise`):

- `weight_decay: 0.1` — catastrophically large for a 400-sample expert. The L2 regularisation gradient dominated the loss gradient; the model converged to `pred ≈ prior`. By epoch 164: val RMSE 10.75 dB vs prior 10.86 dB — only 0.12 dB improvement despite clean loss curve.
- `lambda_recon: 0.0` / `mse_weight: 0.0` — main 513px reconstruction loss was completely switched off; only the lower-resolution multiscale auxiliary was active. Training loss decreased (multiscale converged) while full-res RMSE barely moved.

| Bug | Bad value | Fixed value | Impact |
|-----|-----------|-------------|--------|
| `training.weight_decay` | 0.1 | 0.015 | Catastrophic over-regularisation → pred≈prior |
| `loss.lambda_recon` | 0.0 | 1.0 | 513px reconstruction completely off |
| `loss.loss_type` | `huber` (delta=0.14 norm) | `mse` | Removes threshold masking of small errors |
| `corridor_weighting.enabled` | true (sigma=40) | false | Centre-only weighting degraded edge pixels |
| `prior_residual_path_loss.loss_weight` | 0.0 | 0.5 | No residual supervision term |
| `nlos_focus_loss.enabled` | false | true (weight 0.2) | NLoS pixels under-supervised |

**Speed improvements applied:**

- DataLoader: `num_workers` 4→6, `val_num_workers` 2→4, `persistent_workers: true`, `prefetch_factor: 4`
- `batch_size` 1→2, `gradient_accumulation_steps` 16→8 (same effective batch; faster GPU utilisation)

**SOA training additions:** same SWA, target label noise, and LR-score EMA smoothing as Try 67 (see above).

## Try 69: Try 67 + dual head + corridor training weights + TTA on validation

`Try 69` **forks Try 67** (same HDF5, 3 experts, knife-edge, PDE aux, PMHHNet stem+HF, CutMix guard with FiLM) and implements SOA items that Try 67 had **disabled or YAML-only**:

| Addition | Where |
|----------|--------|
| Dual LoS/NLoS decode | `model.out_channels: 2`, `dual_los_nlos_head.enabled: true`, `_apply_dual_los_nlos_head` in `train_partitioned_pathloss_expert.py` |
| Corridor loss emphasis | `corridor_weighting` in YAML; `_corridor_spatial_multiplier()` multiplies `weighted_mask` for main / multiscale / legacy MSE terms (not applied to `nlos_focus_loss` / PDE by default) |
| D4 TTA on val | `test_time_augmentation.use_in_validation: true` in generated expert YAMLs |

**Folder:** `TFGpractice/TFGSixtyNinthTry69/` via `TFGpractice/scripts/bootstrap_try69_from_try67.py`. **Configs:** `scripts/generate_try69_configs.py` writes `experiments/sixtyninth_try69_experts/try69_expert_*.yaml`. **Cluster:** `run_sixtyninth_try69_*.slurm`, `submit_try69_*`, default `MASTER_PORT` **29976** and submitter `--base-master-port` **30286** to avoid collisions with Try 67/68.

**Checkpoints:** Architecture differs from Try 67 (**two output channels**); do **not** expect load-state compatibility with Try 67 `best_model.pt` without surgery.

### Bug fixes applied to Try 69 (2025-04)

Try 69 inherited the same Try 67 configuration bugs (all three expert YAMLs patched):

| Bug | Bad value | Fixed value |
|-----|-----------|-------------|
| `loss.lambda_recon` | 0.0 | 1.0 |
| `loss.mse_weight` | 0.0 | 1.0 |
| `training.generator_objective` | `full_map_rmse_only` | `legacy` |
| `corridor_weighting.enabled` | true (sigma=40) | false |
| `prior_residual_path_loss.loss_weight` | 0.0 | 0.5 |
| `dropout` | 0.20 | 0.12 |
| `weight_decay` | 0.030 | 0.015 |
| `cutmix_prob` | 0.45 | 0.25 |
| `clip_max_db` per expert | 125 / 135 / 145 | 150 / 160 / 175 |

**Speed improvements:** DataLoader workers 0→6, `val_num_workers` 0→4, `persistent_workers: true`, `prefetch_factor: 4`, `batch_size` 1→2, `gradient_accumulation_steps` 16→8.

**SOA additions:** SWA, target label noise, LR-score EMA smoothing (same as Try 67 — see details there).

## Try 70: Try 68 PMHHNet + multi-scale quad auxiliary heads

`Try 70` keeps the **Try 68** data stack and training loop (`train_partitioned_pathloss_expert.py`) but swaps the generator for **`PMHHNetTry70MultiQuad`** when `model.arch` is `pmhhnet_try70` (aliases: `try70`, `pmhhnet_multiquad`).

| Piece | Where |
|-------|--------|
| Extra heads (257: 4+1, 129: 16+1, 65: 64+1) | `model_try70_multiquad.py`; fused features → tiled + global low-res residuals |
| Main loss | Same Huber + multiscale on 513 as Try 68 `full_map_rmse_only` |
| Aux loss | `try70.aux_loss_weight` × mean Huber over native + bilinear-up global branches |
| Warm-start from Try 68 | `try70.init_checkpoint` (or Slurm `TRY70_INIT_CHECKPOINT` via `prepare_runtime_config.py --try70-init-checkpoint`) — `load_state_dict(..., strict=False)` |
| Val blend diagnostic | `try70.blend_report_first_batch` → `try70_blend_first_batch` in `evaluate_validation` summary (first val batch) |

**Folder:** `TFGpractice/TFGSeventiethTry70/` (full tree copied from Try 68; cluster scripts renamed for Try 70).

**Cluster:** `cluster/run_seventieth_try70_4gpu.slurm`, `cluster/submit_try70_open_sparse_4gpu_sequential.py` (optional `--cancel-all-user-jobs`, default `MASTER_PORT` base **30290**). Upload uses shared `TFGpractice/cluster/upload_and_submit_experiments.py` with `--local-dir TFGSeventiethTry70`. Remote path: `/scratch/nas/3/gmoreno/TFGpractice/TFGSeventiethTry70`.

**Launch (sert, 2026-04-14):** after `scancel -u gmoreno`, repo sync + four chained `sbatch` on `sert-2001` for `try70_expert_open_sparse_lowrise.yaml`: job IDs **10030660** → **10030661** → **10030662** → **10030663** (`afterany` chain). Job 1 exported `TRY70_INIT_CHECKPOINT` = Try 68 `try68_expert_open_sparse_lowrise/best_model.pt`; segments 2–4 resume `outputs/try70_expert_open_sparse_lowrise/best_model.pt`.

### Bug fixes applied to Try 70 (2025-04)

Three bugs found and fixed in `model_try70_multiquad.py` and `train_try70_multiquad.py`:

**Bug 1 — Global-branch double-counting in `try70_auxiliary_loss`:**
The loss loops counted the global branch channel inside the per-tile range, then added it again explicitly:
```python
# BEFORE (wrong): range(5) counted 4 tiles + global channel 4
for k in range(5):  # should be range(4)
    total += _hub(o257[:, k:k+1], ...)
```
Fixed: `range(5/17/65)` → `range(4/16/64)` so the global branch is only counted once via the explicit bilinear-upsample path.

**Bug 2 — Aux heads supervised against full target instead of residual target:**
Aux heads predict the **residual** (same space as the main head: `pred − prior`). The loss was comparing them to the full path-loss `target`, not `target − prior`, inflating the loss by the mean prior value (~80–120 dB normalized). Fixed by adding `prior: Optional[torch.Tensor]` parameter to `try70_auxiliary_loss`; internally uses `res_target = target − prior` for all auxiliary comparisons.

**Bug 3 — Blend search ran on first batch only:**
`_validate_rmse` called `try70_blend_search_rmse_physical` on a single validation sample (first batch). This gave noisy alpha/channel rankings driven by one city. Fixed: SSE is now accumulated per `(channel, alpha)` across the **full validation set**; RMSE is computed from the aggregated SSE at the end. Alpha sweep [0.0 … 1.0] in configurable steps (`try70.blend_alpha_steps`, default 11). Results sorted by `best_rmse_phys` ascending in the JSON report.

## Try 71: Try 68 + heteroscedastic uncertainty (Kendall & Gal 2017)

`Try 71` forks Try 68's `open_sparse_lowrise` expert to add **per-pixel aleatoric uncertainty estimation** using the heteroscedastic regression formulation of Kendall & Gal (NeurIPS 2017).

**Core idea:** Instead of predicting a single mean residual, the model outputs two channels:
- channel 0: mean residual `μ` (same as Try 68)
- channel 1: log-variance `log σ²` (new)

The NLL loss is:
```
L_NLL = (μ − y)² / exp(log σ²) + log σ²
```

This automatically down-weights pixels where the model is uncertain (high `σ²`), which in practice corresponds to deep-NLoS / hard-diffraction pixels where ray-traced ground truth itself has high variance. The model learns **where not to be confident** rather than forcing uniform fit across all pixels.

**Evaluation: RMSE-vs-coverage curve**

At inference the per-pixel uncertainty σ is used as a confidence score. The evaluation sweeps confidence thresholds τ ∈ [0, ∞) and reports:
- `coverage`: fraction of ground pixels with σ ≤ τ
- `rmse_dB`: RMSE on the covered subset

This produces a selective-prediction curve: at 100% coverage the RMSE is the full-map result; as τ decreases the model only commits to easy pixels and RMSE drops. The gap between full-coverage RMSE and 50%-coverage RMSE measures how much uncertainty the model has captured.

Output: both the JSON metrics file and the per-epoch JSONL include a `coverage_rmse_curve` list `[{tau, coverage, rmse_phys}]` at 20 threshold steps.

**Architecture change:**
- `model.out_channels: 2` in YAML
- `PMHHNetResidualRegressor` with `out_channels=2` already outputs the correct shape; channel 0 = mean, channel 1 = log_var
- `log_var` is clamped to `[−10, 10]` to avoid numerical explosion

**Loss change (in `train_partitioned_pathloss_expert.py`):**
```python
mu   = pred[:, 0:1]
lv   = pred[:, 1:2].clamp(-10, 10)
nll  = (mu - target)**2 / lv.exp() + lv          # Kendall & Gal NLL
loss = (nll * mask).sum() / mask.sum().clamp_min(1)
```
Multiscale and NLoS focus losses are computed on `mu` only (not `lv`).

**Folder:** `TFGpractice/TFGSeventyFirstTry71/` — generated from Try 68 via `TFGpractice/scripts/bootstrap_try71_from_try68.py`.

**Scope:** `open_sparse_lowrise` only (mirrors the Try 68 checkpoint available on the cluster). Cluster script copies `outputs/try68_expert_open_sparse_lowrise/best_model.pt` into `outputs/try71_expert_open_sparse_lowrise/` before the first epoch; `load_state_dict(strict=False)` loads all shared weights (new log_var head initialises randomly).

**Bootstrap script:** `TFGpractice/scripts/bootstrap_try71_from_try68.py`

## Try 72: Try 68 + sparse receivers + smaller model + larger batch

`Try 72` copies the **Try 68** training stack (`train_partitioned_pathloss_expert.py`, six topology experts, PMHHNet, prior + residual, same data channels and splits) and adds **receiver subsampling** to stress generalisation under **fewer supervised pixels per map** (closer in spirit to sparse drive-test links than “every raster cell always counts”).

### Receiver subsample (`data.receiver_subsample`)

| Key | Role |
|-----|------|
| `enabled` | If true, loss and primary val metrics use a **sparse** mask. |
| `keep_fraction` | Target fraction of **original** valid ground pixels kept (after tile cap), default **0.01** (1%). |
| `tile_side_m` | Tile size in metres for the per-area cap (default 1000). |
| `max_rx_per_tile` | Max receivers sampled per tile (default 32). |
| `val_seed` | Base for deterministic val subsampling (+ internal offset per batch index). |

Implementation: `data_utils.apply_receiver_subsample_mask` (numpy grouping per sample). Train multiplies the loss mask by a **new** gate each step (`seed = cfg.seed + epoch×1_000_003 + step×47`). When enabled, JSON includes **`metrics.path_loss_dense_reference`** (RMSE on **dense** valid pixels) alongside sparse `metrics.path_loss`.

### Model / batch (vs Try 68 defaults in repo YAML)

- `base_channels: 20`, `hf_channels: 10` (all six experts).
- `training.batch_size: 4`, `data.val_batch_size: 4`, `training.gradient_accumulation_steps: 4`.

**Folder:** `TFGpractice/TFGSeventySecondTry72/`. **Docs:** `README.md` (overview + config pointers). **Plots:** `scripts/plot_try72_metrics.py` (val sparse vs dense reference on panel 0).

## Related documents

- `SUPERVISOR_SUMMARY_TRIES_20_TO_32.md`
- `PAPER_SOURCES_TRIES_20_TO_32.md`
- `TRY29_VISUAL_REVIEW_AND_RADIAL_PLAN.md`
- `TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md`
- `PATH_LOSS_PRIORITY_NEXT_STEPS.md`
- `TRIES_33_TO_36_PHYSICAL_PRIORS_AND_BUILDING_MASK.md`
- `TRIES_37_TO_40_NEW_DATASET_AND_MASK_VERIFICATION.md`
- `TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md`
- `analysis/formula_prior_generalization_try41.md`
- `FORMULA_PRIOR_CALIBRATION_SYSTEM.md`
- `FOR_GENIA_SUMMARY_TRIES_20_TO_42.md`
- `PAPER_SOURCES_TRIES_20_TO_42.md`
- `TRY42_SOURCES_AND_PMNET_SCHEMA.md`
- `TRY42_PMNET_RESIDUAL_ARCHITECTURE.md`
- `FOR_GENIA_SUMMARY_TRIES_20_TO_44.md`
- `PAPER_SOURCES_TRIES_20_TO_44.md`
- `TRY43_TRY44_PMNET_CONTROLS.md`
