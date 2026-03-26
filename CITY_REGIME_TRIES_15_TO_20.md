# City-Regime Tries 15 to 20

These folders turn the `NEXT_TRIES_CITY_REGIME.md` plan into runnable repo scaffolding.

## What is already prepared

- Dedicated folders:
  - `TFGFifteenthTry15`
  - `TFGSixteenthTry16`
  - `TFGSeventeenthTry17`
  - `TFGEighteenthTry18`
  - `TFGNineteenthTry19`
  - `TFGTwentiethTry20`
- One YAML per try under `experiments/...`
- One 1-GPU Slurm launcher per try
- Support in `cluster/upload_and_submit_experiments.py`

## Important scope note

The new folders are intentionally based on the stable FiLM + antenna-height pipeline from Try 14 so they can run immediately.
That means the repo scaffolding for Try 15 to 20 is ready now, but the more ambitious backbone changes from the design note
(attention bottlenecks, mixture of experts, 3D-aware variants, and physics-aware losses) still need explicit code changes
before those tries become architecturally distinct.

## Why the first submitted configs were changed

Before submitting the jobs, the prediction panels already exported in `D:\Dataset_Imagenes` were checked visually.
The goal was not to guess blindly from RMSE alone, but to see what kind of error the current family was making.

### What looked wrong in the images

The most relevant patterns seen in the `gt_pred` panels were:

- Checkerboard / grid-like artifacts in `path_loss` predictions.
- Loss of thin geometric structure around streets, block edges, and occlusion transitions.
- Over-textured predictions: the output looked more like a stylized image than a smooth physical field.
- Local bias patches: some regions were systematically too dark or too bright relative to the GT.
- In `delay_spread`, predictions tended to collapse contrast and become too flat.

These visual patterns matter because a physically plausible path-loss map should usually be smoother than a GAN-textured image while still respecting strong geometry boundaries.

## Interpretation of those failures

### 1. BatchNorm was suspicious with batch size 1

All these tries run with `batch_size: 1`. In that regime, `batch norm` often becomes unstable or too sample-specific.
That can show up as:

- inconsistent local contrast,
- patchy intensity drift,
- checkerboard-looking structure after upsampling blocks,
- worse generalization than expected even if training loss falls.

Because of that, the configs were changed from `batch` to `group` normalization in the generator and discriminator for the launched tries.

### 2. The adversarial term was probably over-helping texture and under-helping physics

The visual artifacts did not look like simple blur only. They looked like the model was trying to synthesize plausible texture.
That is often useful for natural images, but here the target is a propagation field.

For this problem, too much GAN pressure can:

- create fake high-frequency detail,
- improve local visual sharpness while hurting physical RMSE,
- make building shadows and street transitions look plausible but numerically wrong.

Because of that:

- some tries were switched to pure regression (`lambda_gan: 0.0`),
- some kept only a very small GAN weight (`0.003` or `0.005`) as a controlled comparison,
- the family now tests whether the GAN is helping structure or only adding artifacts.

### 3. The model needed a stronger geometric cue

The GT maps clearly contain radial / distance-dependent behavior around the transmitter plus occlusion effects from the topology.
When the prediction misses that, one likely issue is that the network is relying too much on texture in the topology image and not enough on explicit geometric distance.

That is why `distance_map_channel: true` was enabled in tries 16 to 19:

- it gives the network direct access to radial distance,
- it should help recover large-scale attenuation trends,
- it reduces the need for the CNN to infer distance only from implicit context.

### 4. Smoothing should be tuned, not assumed

Some panels suggested that the raw prediction contains either noise or overly sharp false detail.
But too much smoothing would also erase street/building boundaries.

Because of that, the postprocess kernel candidates were widened in several tries:

- `[1, 3]` for the more conservative runs,
- `[1, 3, 5]` for the more regularized runs.

This keeps the comparison measurable instead of hard-coding a single smoothing strength.

## Why each try was adjusted the way it was

### Try 15

`TFGFifteenthTry15` was turned into the clean baseline:

- `group norm`,
- no GAN,
- no distance map,
- slightly lower generator LR.

Reason:
start from the simplest version that is least likely to hallucinate texture. This gives a reference for whether the GAN and the distance map are actually helping.

### Try 16

`TFGSixteenthTry16` adds distance information but still removes GAN pressure:

- `group norm`,
- `distance_map_channel: true`,
- `lambda_gan: 0.0`.

Reason:
if this beats Try 15, then explicit geometry is helping more than adversarial detail.

### Try 17

`TFGSeventeenthTry17` is the light-GAN comparison:

- `group norm`,
- `distance_map_channel: true`,
- `lambda_gan: 0.003`.

Reason:
test whether a very small adversarial term can keep boundaries a bit cleaner without reintroducing the strong texture artifacts.

### Try 18

`TFGEighteenthTry18` is the moderate-GAN comparison:

- `group norm`,
- `distance_map_channel: true`,
- `lambda_gan: 0.005`,
- stronger smoothing candidates.

Reason:
check whether a mid-point between pure regression and the old GAN-heavy behavior is a better tradeoff.

### Try 19

`TFGNineteenthTry19` is the most conservative regularized regression:

- `group norm`,
- `distance_map_channel: true`,
- `lambda_gan: 0.0`,
- lower generator LR,
- slightly higher weight decay,
- broader smoothing candidates with default kernel `3`.

Reason:
this is the “reduce artifacts first” run. It is the one most directly aimed at making the output field numerically stable and less textured.

## What was not claimed

These changes were made because they are defensible given the observed image failures, not because they guarantee improvement.

They do **not** guarantee:

- a 100% better result,
- a lower RMSE in every regime,
- that the best try will be the most regularized one.

They **do** make the experiment family more informative, because the comparison now isolates three main hypotheses:

1. Is BatchNorm hurting because batch size is 1?
2. Is the GAN creating false texture?
3. Does explicit distance help recover large-scale path-loss structure?

## Practical conclusion

The launched tries are not random variants.
They are a controlled response to specific visual failure modes already present in the exported dataset predictions.

If the resulting metrics improve and the new prediction panels lose the checkerboard / fake-texture look, that will be evidence that:

- `group norm` was the right change,
- weaker or zero GAN was beneficial for this task,
- explicit geometric distance is useful.

If they do not improve, the next step should not be “more of the same”, but a deeper model change:

- better upsampling,
- residual / dilated blocks,
- attention only after the artifact issue is under control,
- or a more explicit physics-aware loss beyond the current config-only changes.

## Recommended usage

- Use **Try 15** as the first runnable baseline of the new family.
- Use **Try 16 to 20** as isolated branches/folders to evolve that baseline without contaminating previous tries.
- Keep `CKM_180326_antenna_height.h5` as the source for antenna height to avoid the all-zero height problem noted in the day-2 docs.
- The separation between common changes and true new-try changes is documented in `TRIES_COMMON_CHANGES_AND_NEXT_TRIES.md`.

## Example uploads

```powershell
cd C:\TFG\TFGpractice
python cluster\upload_and_submit_experiments.py --preset fifteenth
python cluster\upload_and_submit_experiments.py --preset sixteenth
```
