# SOA — Papers to track and implementation roadmap

Companion to `TFGSixtySeventhTry67/SOA_COMPARISON.tex` (thesis narrative) and `SOA_IMPLEMENTATION_STATUS.md` per try.  
This file lists **additional** recent work not yet fully folded into the comparison tables, with a **concrete fit** to the CKM project (513², city holdout, continuous UAV height, building-masked loss).

Conventions: **S** = small change (loss/aug/channel), **M** = new module or second stage, **L** = new backbone or multi-week effort.

---

## A. Radio-map / path-loss prediction (closest match)

| Work | Identifier | Core idea | Fit to this TFG | Horizon |
|------|----------------|-----------|-----------------|-----------|
| **RadioTransformer** | [arXiv:2501.05190](https://arxiv.org/abs/2501.05190) | Hybrid transformer–CNN encoder + conv decoder for radio maps | Strong baseline for *same-distribution* outdoor maps; **height is not** the main axis — adapt as encoder replacement or cross-attention to height embedding | **L** |
| **RadioMamba** | [arXiv:2508.09140](https://arxiv.org/abs/2508.09140) | Mamba–UNet hybrid, linear global context vs. full self-attention | Good **efficiency vs. accuracy** story for 513²; needs new stack + careful DDP | **L** |
| **PathFinder** | [arXiv:2512.14150](https://arxiv.org/abs/2512.14150) | Disentangled building/Tx features, mask-guided low-rank attention, **Transmitter-Oriented Mixup**, S2MT benchmark | Single-UAV Tx matches only partly; **Mixup + mask-guided attention** transferable to NLoS tails and **multi-height** as “virtual Tx” mixup | **M** |
| **Reciprocity-aware CNNs** | [arXiv:2504.03625](https://arxiv.org/abs/2504.03625) | Synthetic aug from **downlink-only** data to generalise uplink/downlink | No drive-test here, but idea = **physics-aware augmentation** (e.g. swap Tx/Rx height roles, mirror link budget) for **height OOD** | **S**–**M** |
| **Effective outdoor path loss (corridor weighting)** | Gao et al., [arXiv:2601.08436](https://arxiv.org/abs/2601.08436) | Multi-layer segmentation + **Tx–Rx corridor** loss emphasis | Try **69** implements a **radial Tx-centred** proxy; full per-Rx corridor is **S**–**M** in `data_utils` | **S**–**M** |

---

## B. Physics-informed / generative (thesis “future work” row)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **ReVeal** (PINN-style residuals) | [arXiv:2502.19646](https://arxiv.org/abs/2502.19646) | Helmholtz / wave residual on field | Try 67+ already use **masked Laplacian** as light PDE prior; full operator = **L** | **L** |
| **RadioDiff** (diffusion refinement) | Cited in internal SOA (section 7) | Diffusion refiner on top of CNN map | Second stage after PMHHNet; **M**–**L**, heavy sampling cost at inference | **M**–**L** |
| **Neural operators / Fourier** | Various (FNO, AFNO on 2D maps) | Learn solution operator across resolutions or parameters | Could condition on **height as branch**; data cost high | **L** |

---

## C. Generalisation / adaptation (city holdout angle)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **Test-time adaptation** (e.g. RadioPiT-style) | See SOA fair-comparison “No” row | Few gradient steps at test on unlabeled target city | Directly targets **city holdout** gap; needs unlabeled target maps + stability guardrails | **M**–**L** |
| **Foundation / multi-city pretrain** (FM-RME-style) | SOA (section 7) | Pretrain encoder on many cities, finetune per expert | Aligns with **66 cities** but needs **train protocol** and storage | **L** |

---

## D. Suggested priority for *next* code experiments

1. **PathFinder-style Transmitter-Oriented Mixup** (or height-conditioned mixup) — **M**, targets NLoS / distribution shift without new backbone.  
2. **Reciprocity / link-symmetric augmentations** on `topology` + scalar height — **S**, low risk.  
3. **Full Gao corridor** (per-pixel weight along Tx→Rx ray, not only radial from centre) — **M**, improves SOA faithfulness.  
4. **AIRMap-style few-shot calibration** (small held-out slice of target city, no full retrain) — **M**, aligns with “worse than prior” on strict holdout if you allow **ethical leakage** only for ablation.  
5. **NeWRF / GRaF-style sparse-measurement head** (auxiliary branch predicting from sparse probes) — **L**, different data model than full ray-traced grid.  
6. **RadioMamba or slim transformer** encoder block inside PMHHNet — **L**, after ablations on 1 GPU.

---

## E. Digital twins and *fast* surrogate simulators (latency story)

| Work | Identifier | Core idea | Fit to this TFG | Horizon |
|------|----------------|-----------|-----------------|-----------|
| **AIRMap** | [arXiv:2511.05522](https://arxiv.org/abs/2511.05522) | Single-input **U-Net** on **2D elevation / building heights**; ms inference vs. ray trace; **light calibration** with ~20% field samples | Same *inputs family* as CKM (heightmap); **no continuous UAV height** in the public pitch — compare as **“speed vs. ray trace”** baseline; calibration idea = **test-time** or **few-shot adapter** | **M**–**L** |

**Caveat:** AIRMap numbers are on their Boston-scale dataset and metrics (path gain, RMSE ~4 dB claimed); **not** directly comparable to city-holdout CKM + continuous height.

---

## F. Indoor / challenge leaders (geometry + multi-frequency)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **TransPathNet** | [arXiv:2501.16023](https://arxiv.org/abs/2501.16023) | **Two-stage**: transformer feature extraction + multiscale conv attention decoder; ICASSP 2025 indoor challenge | Borrow **stage-2 refiner** or **multiscale attention decoder** ideas for NLoS edges; indoor ≠ outdoor | **M**–**L** |
| **IPP-Net** (SIA, ICASSP 2025) | See challenge leaderboard / paper trail | U-Net on RT data, **9.5 dB** class weighted RMSE indoor | Decoder width / loss weighting patterns; frequency conditioning analogous to your **per-expert clamps** | **S**–**M** |

Use these mainly for **thesis positioning** (“indoor SOTA RMSE is ~9–10 dB under challenge rules”) rather than dataset-merge.

---

## G. UAV cellular — classical ML on *tabular* features (not 2D maps)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **Triple-layer ML for cellular UAV** | [arXiv:2505.19478](https://arxiv.org/abs/2505.19478) | STW + bagged trees + **GPR** stack on **3D/2D distances, azimuth, elevation** from drive tests | **No raster map**: features are closer to your **scalar `antenna_height_m` + geometry statistics**; useful as **second baseline** (“non-CNN KPI predictor”) in thesis discussion, not drop-in for 513² | **S** (analysis) / **M** if you build a **tabular head** fused with CNN |

---

## H. Neural *fields* for RF (continuous position / spectrum)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **NeWRF** | [arXiv:2403.03241](https://arxiv.org/abs/2403.03241) / [ICML 2024](https://proceedings.mlr.press/v235/lu24j.html) | NeRF-style **wireless radiation field** from **sparse** samples; predicts channel at unvisited locations | Aligns with **“few measurements + digital twin”** narrative; training pipeline ≠ HDF5 full maps unless you subsample pixels as rays | **L** |
| **GRaF** (generalizable RF radiance fields) | [arXiv:2502.05708](https://arxiv.org/abs/2502.05708) | **Cross-scene** generalization for spatial spectra; geometry-aware transformer + neural ray tracing | Closest neural-field line to **city holdout**; heavy implementation | **L** |
| **NeRF²** (MobiCom 2023) | [PolyU page / PDF](https://web.comp.polyu.edu.hk/csyanglei/data/files/nerf2-mobicom23.pdf) | RF radiance fields for **material + multipath** | Mostly indoor / spectrum imaging angle; cite as **historical bridge** to NeWRF/GRaF | **L** (research) |

---

## I. Measurement-based UAV channels (physics priors, not DL maps)

| Work | Identifier | Core idea | Fit | Horizon |
|------|----------------|-----------|-----|---------|
| **UAV A2G measurements 1 & 4 GHz** | [arXiv:2501.17303](https://arxiv.org/abs/2501.17303) | Large-scale / small-scale parameters vs. altitude, LoS/NLoS | Supports **FiLM + elevation map** design rationale; use for **cite + ablation** on height bins | **S** (doc) |

---

## J. Curated list (repos)

- [Awesome Radio Map (categorized)](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized) — keeps RadioMamba, RadioDiff, RadioUNet lineages updated.
- [RadioMap Challenge (results)](https://radiomapchallenge.github.io/results.html) — PMNet / RMTransformer / late submissions (outdoor + indoor tracks).

When a row moves from “Missing” to “Implemented”, update **`SOA_IMPLEMENTATION_STATUS.md`** in the active try folder and keep **`VERSIONS.md`** in sync for the thesis timeline.
