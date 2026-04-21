# LoS Model Shortlist

Date: 2026-04-21

## Stored local checkpoints

Ranked by the best locally available LoS validation evidence, with scope noted:

1. `try74_expert_band4555_allcity_los`
   - Stored val RMSE: `3.6758 dB`
   - Scope: `allcity_los`, but only height band `45-55 m`
   - Source: `cluster_outputs/TFGSeventyFourthTry74/try74_expert_band4555_allcity_los/validate_metrics_latest.json`
2. `try75_expert_allcity_los_small` (`best_model.pt` = epoch 1)
   - Stored val RMSE: `3.7931 dB`
   - Scope: `allcity_los`, all validation heights
   - Source: `cluster_outputs/TFGSeventyFifthTry75/try75_expert_allcity_los_small/validate_metrics_epoch_1.json`
3. `try76_expert_dense_block_midrise_los`
   - Best val RMSE from history: `3.7429 dB` at epoch `43`
   - Scope: topology-specific `dense_block_midrise`
   - Source: `cluster_outputs/TFGSeventySixthTry76/try76_expert_dense_block_midrise_los/history.json`
4. `try76_expert_mixed_compact_lowrise_los`
   - Best val RMSE from history: `3.9502 dB` at epoch `33`
   - Scope: topology-specific `mixed_compact_lowrise`
   - Source: `cluster_outputs/TFGSeventySixthTry76/try76_expert_mixed_compact_lowrise_los/history.json`
5. `try76_expert_open_sparse_lowrise_los`
   - Best val RMSE from history: `4.3356 dB` at epoch `6`
   - Scope: topology-specific `open_sparse_lowrise`
   - Source: `cluster_outputs/TFGSeventySixthTry76/try76_expert_open_sparse_lowrise_los/history.json`
6. `try76_expert_mixed_compact_midrise_los`
   - Best val RMSE from history: `4.5463 dB` at epoch `34`
   - Scope: topology-specific `mixed_compact_midrise`
   - Source: `cluster_outputs/TFGSeventySixthTry76/try76_expert_mixed_compact_midrise_los/history.json`
7. `try76_expert_open_sparse_vertical_los`
   - Best val RMSE from history: `4.7252 dB` at epoch `25`
   - Scope: topology-specific `open_sparse_vertical`
   - Source: `cluster_outputs/TFGSeventySixthTry76/try76_expert_open_sparse_vertical_los/history.json`

## Chosen model for full validation

Chosen candidate: `try75_expert_allcity_los_small/best_model.pt`

Reason:
- `best_model.pt` is the healthy epoch-1 checkpoint
- it is trained for `allcity_los`
- it covers all validation heights, unlike the narrower Try74 `45-55 m` run

## Full validation on DirectML

Command used:

```powershell
C:\TFG\.venv\Scripts\python TFGpractice\tmp_review\histograms_try74\evaluate_single_checkpoint_fullsplit.py `
  --try-root C:\TFG\TFGpractice\TFGSeventyFifthTry75 `
  --config C:\TFG\TFGpractice\TFGSeventyFifthTry75\experiments\seventyfifth_try75_experts\try75_expert_allcity_los.yaml `
  --checkpoint C:\TFG\TFGpractice\cluster_outputs\TFGSeventyFifthTry75\try75_expert_allcity_los_small\best_model.pt `
  --device directml `
  --split val `
  --batch-size 2 `
  --num-workers 0 `
  --progress-every 100 `
  --save-json C:\TFG\TFGpractice\tmp_review\histograms_try74\try75_epoch1_allcity_los_full_val_directml.json
```

Result on the full validation split (`2750` samples):

- Ground RMSE: `4.7887 dB`
- LoS RMSE: `3.7956 dB`
- NLoS RMSE: `11.3023 dB`
- LoS valid pixels: `382,752,235`
- NLoS valid pixels: `31,131,895`

Per `city_type_3` LoS RMSE:

- `dense_highrise`: `3.6474 dB`
- `mixed_midrise`: `3.7490 dB`
- `open_lowrise`: `3.8577 dB`

Saved summary:

- `tmp_review/histograms_try74/try75_epoch1_allcity_los_full_val_directml.json`
