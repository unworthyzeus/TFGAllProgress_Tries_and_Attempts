# Next Tries: City-Regime Path Loss Experiments

Target: move path-loss RMSE from the current ~19 dB plateau toward the 5 dB objective.
All new tries in this family are planned as one-GPU runs.

## Current Problem

- The mixed path-loss model is stuck around ~19 dB RMSE.
- More compute did not break the plateau.
- LoS and nLoS do not fail in the same way, so a single undifferentiated model is wasting capacity.
- Antenna height only helps if the input is actually populated with real values.

## What the documentation says

- The repo history shows that direct MSE in dB got the most stable improvement among the early tries, but it still plateaued near ~19.7 dB.
- The day-2 notes explicitly suggest LoS-only and NLoS-only runs, plus height normalization experiments.
- The paper notes in `TFG_Proto1` say standard 2D U-Nets are limited on long-range reflections and that stronger spatial inductive bias is needed.
- The papers also argue that city geometry matters, which supports grouping cities by type instead of training one model per city.
- The exported visual pipeline already exists for `D:/Dataset_Imagenes`, so the next experiments can be compared visually as well as by RMSE.

## Why not one model per city

- The data is too sparse to spend one model on every city.
- The repo docs point to shared propagation structure inside city families.
- A city-type model can reuse signal from similar environments while still specializing enough to matter.
- That keeps the experiment count manageable and makes comparisons cleaner.

## What the current results say

- More compute alone did not move the mixed model off the plateau.
- Antenna height helped only if the height channel is real and not zero-filled.
- LoS and nLoS behave like different tasks.
- The hard part is nLoS and long-range geometry, not the easy LoS regime.

## What we should stop doing first

- One model per city.
- Bigger 2D U-Nets without a new inductive bias.
- Re-running the same mixed LoS/nLoS setup with only a different learning rate.

## Proposed city buckets

Use one model per city type, not one model per city.

- Small city: sparse, lower building density, lower average height.
- Medium city: mixed density and mixed height distribution.
- Big city: dense urban fabric, taller buildings, more occlusion.
- Optional high-rise bucket if the data clearly separates it from the other three.

The bucket can come from a simple offline classifier built from city statistics such as building density, mean height, height variance, and LoS ratio.

## Proposed try series

### Try 15 - city-conditioned shared baseline

- Backbone: stronger U-Net / ResUNet baseline.
- Conditioning: city-type embedding plus antenna height.
- Heads: separate LoS and nLoS heads, or a gated output head.
- Purpose: measure whether city-type conditioning helps before changing the backbone.

### Try 16 - long-range context backbone

- Backbone: dilated U-Net or residual U-Net with a larger receptive field.
- Conditioning: same as Try 15.
- Purpose: test whether long-range reflections and occlusions are the main bottleneck.

### Try 17 - attention bottleneck

- Backbone: U-Net with attention or a transformer bottleneck.
- Conditioning: city-type embedding plus antenna height.
- Purpose: compare attention-based global context against plain convolution.

### Try 18 - mixture of experts

- Backbone: shared encoder with experts per city bucket.
- Router: city-type gate, optionally helped by LoS dominance.
- Purpose: let the model specialize without exploding into one model per city.

### Try 19 - physics-aware best backbone

- Backbone: whichever of Try 15 to 18 wins.
- Extra loss: spatial smoothness or propagation-aware regularization.
- Optional confidence head: predict when to trust the network output.
- Purpose: use physics only after the architecture is already better.

### Try 20 - 3D-aware variant if the data supports it

- Backbone: 3D or volumetric model, or a slice-aware model with vertical context.
- Purpose: use height structure explicitly instead of only as a scalar input.

## Recommended order

1. Try 15 as the baseline for the new family.
2. Try 16 if Try 15 is still limited by context.
3. Try 17 if long-range dependency is still missing.
4. Try 18 if city buckets are clearly separable.
5. Try 19 only after one of the above is actually better.
6. Try 20 only if the data pipeline can support it cleanly.

## Upload script

Use the consolidated uploader instead of maintaining one script per try:

```powershell
python cluster\upload_and_submit_experiments.py --preset ninth --gpus 1
python cluster\upload_and_submit_experiments.py --preset thirteenth --gpus 1
python cluster\upload_and_submit_experiments.py --local-dir TFGCityRegimeTry15 --slurm cluster/run_cityregime_try15_1gpu.slurm --gpus 1
```

Example for a future try folder that lives in the repo root:

```powershell
python cluster\upload_and_submit_experiments.py `
	--local-dir TFGCityRegimeTry16 `
	--slurm cluster/run_cityregime_try16_1gpu.slurm `
	--gpus 1
```

Example for a pure upload without sbatch submission:

```powershell
python cluster\upload_and_submit_experiments.py --preset ninth --gpus 1 --upload-only
```

The script also accepts explicit `--local-dir` and one or more `--slurm` paths, so the same entry point can handle the current tries and the future city-regime experiments.

## What to compare in the plots

- Mixed validation RMSE.
- LoS-only RMSE.
- NLoS-only RMSE.
- City-bucket RMSE.
- Generalization gap between train and validation.

If a new try improves only LoS and leaves nLoS unchanged, it is not the right direction for the 5 dB target.

## Next Steps

1. Build Try 15 as a real folder with a one-GPU Slurm file and a city-conditioned baseline config.
2. Add Try 16 with a stronger long-range backbone.
3. Keep the same export and comparison pipeline so the RMSE curves and image panels stay comparable.
4. Split the evaluation by LoS, nLoS, and city bucket before deciding whether the next architecture is worth scaling.