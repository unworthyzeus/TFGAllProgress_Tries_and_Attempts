# TRY49 Prior Improvements

## Goal
Improve prior-guided learning without touching Try48 by introducing prior confidence-aware conditioning and lightweight local-CUDA validation.

## Scope
- Base folder: `TFGFortyNinthTry49/` (clone of Try48).
- Keep Try48 unchanged.

## Implemented (v1)
1. Added optional prior confidence channel in `TFGFortyNinthTry49/data_utils.py`.
2. Prior confidence is computed heuristically from:
- LoS probability map
- Distance map
- Local obstruction density
3. Added config switch support:
- `data.path_loss_formula_input.include_confidence_channel: true|false`
- `data.path_loss_formula_input.confidence_kernel_size: int`
4. Updated input channel counting logic so model channels remain consistent.
5. Added Try49 A/B Stage1 configs:
- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen96_baseline.yaml`
- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen96_priorconf.yaml`
6. Added Try49 Stage1 4-GPU slurm launchers:
- `TFGFortyNinthTry49/cluster/run_fortyninthtry49_pmnet_prior_stage1_baseline_4gpu.slurm`
- `TFGFortyNinthTry49/cluster/run_fortyninthtry49_pmnet_prior_stage1_priorconf_4gpu.slurm`
7. Added local prior smoke-test script:
- `TFGFortyNinthTry49/scripts/smoke_test_prior_try49.py`
8. See the end of this document for possible improvements that we'll need to check if are needed after stage 2.

## Intended Effect
- Give model spatial awareness of prior reliability.
- Encourage stronger corrections in low-confidence prior regions (typically harder NLoS/obstructed areas).

## Next Planned Experiments
1. Stage1 (residual) A/B on Try49:
- A: baseline formula prior only.
- B: formula prior + confidence channel.
2. After stage1 converges, train a second model as a post-stage1 tail refiner, not a GAN.
3. Compare:
- path_loss.rmse_physical
- LoS vs NLoS delta
- regime-level errors

## Notes
- Dataset already uses variable transmitter height (`antenna_height_m` from `uav_height`) per sample.
- Try49 is now the only target for these prior improvements.

## Local Validation (completed)
- Command run:
	- `py scripts/smoke_test_prior_try49.py --config experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen96_priorconf.yaml --split train --sample-index 0`
- Result:
	- PASS on synthetic fallback path (local HDF5 file unavailable on this machine).
	- Verified config channel count = 6 with confidence enabled.
	- Verified prior-confidence helper output shape/range sanity.
	- Verified PMNet forward output shape `(1, 1, 513, 513)` with finite values.

## Current Try49 Status
- Stage 1 has been moved to the widened 112-channel setup.
- Current active chain:
	- `10019544`: the old widened-128 run timed out.
	- `10019545`: the old dependent resume run failed with DDP OOM.
- The current replacement chain is the 112-channel stage1 flow.
- The current 112-chain submission IDs are:
	- `10019593` init job
	- `10019594` first resume
	- `10019595` second resume
- The adapted checkpoint is stored at:
	- `TFGFortyNinthTry49/outputs/fortyninthtry49_pmnet_prior_stage1_t49_stage1_w112_4gpu/reduced_try49_epoch30_128_to_112.pt`
- The replacement job resumes from the latest saved checkpoint for the widened-128 run, which is the epoch-30 checkpoint; the run had validation metrics through epoch 32, but no epoch-32 checkpoint was saved.
- The active model is still Try49-started, now reduced to 112 channels for the restart.
- Stage 2 has been rewritten as a separate HDF5-backed tail refiner.
- The old GAN-style stage2 path is no longer the default; stage1 predictions are exported first and stage2 trains from those frozen outputs.
- The obsolete resume jobs `10019594` and `10019595` were cancelled once the new stage2 path was introduced.

## Prior Calibration Result
- Completed calibration output for Try49:
	- `overall`: RMSE `23.5742 dB`, MAE `12.2636 dB`
	- `LoS`: RMSE `3.8137 dB`, MAE `3.0449 dB`
	- `NLoS`: RMSE `41.2678 dB`, MAE `31.8064 dB`
- This is kept here so the current Try49 state and calibration history live in one place.

## Why RMSE Is So Much Larger Than MAE
The gap between RMSE and MAE is the signature of a heavy-tailed error distribution. MAE measures the average absolute miss, so it is sensitive to the typical case. RMSE squares the error before averaging, so a small number of very bad predictions can dominate the metric.

That is exactly what the epoch-32 JSON shows:
- The overall MAE is about `7.76 dB`, which means the model is often in a reasonably controlled error band.
- The overall RMSE is about `19.07 dB`, which means some predictions are much worse than the average case.
- The worst buckets are the NLoS and low-antenna regimes, especially the prior-aware NLoS groups. Those are the outliers that push RMSE up.

This is not a contradiction. It means the model is getting a lot of samples roughly right, but it still fails badly on a smaller subset of hard cases.

## What That Means Practically
This pattern suggests the problem is not only “the model is weak everywhere.” It is more specifically:
- A large fraction of the dataset is already predictable enough that the average error is moderate.
- A smaller fraction of the dataset contains very hard examples where the prior is misleading or the geometry is genuinely ambiguous.
- Those hard examples are the ones that inflate RMSE much more than MAE.

So the right next step is not to make the first model broader in the hope that it fixes everything uniformly. The more sensible path is to keep a strong stage1 model and add a second model that is explicitly designed to reduce tail errors.

## Should The Two Models Be Trained Together?
Short answer: usually no, not for this problem.

Training both large models at the same time makes the optimization problem harder for no clear benefit:
- The first model and the second model start co-adapting to each other before either one has stabilized.
- The second model can end up learning to compensate for noise in the first model instead of learning a clean correction pattern.
- Validation becomes harder to interpret because you no longer know whether an improvement came from the base predictor, the refiner, or an unstable interaction between the two.
- Memory and throughput pressure increase immediately, which is exactly the constraint that already caused issues with wider Try49 runs.

For this dataset and this metric pattern, sequential training is the better default.

## Preferred Strategy: Train After Stage1
The strongest practical option is to train the second model after stage1 has converged, not concurrently.

The rationale is simple:
- Stage1 should learn the stable, general predictor.
- Stage2 should only see the residual failure modes left by stage1.
- Once stage1 is fixed, stage2 can be specialized for the tail without destabilizing the baseline.

This gives a cleaner decomposition:
- Stage1 handles the broad structure of the propagation problem.
- Stage2 handles difficult outliers, usually NLoS and low-confidence regimes.

That is a much better fit than a jointly trained two-model system.

## What The Second Model Should Actually Do
The second model should not be a generic “more of the same” regressor. Its job should be narrowly defined:
- It should learn a correction to the stage1 output, not a full replacement.
- It should be conservative and only act when the first model is likely wrong.
- It should be allowed to do nothing on easy cases.

The safest form is a residual tail-refiner:
- Input: original features plus stage1 prediction plus confidence or uncertainty signals.
- Output: a bounded correction term or a gate on whether to apply a correction.
- Final prediction: stage1 output plus a clipped or weighted delta.

That keeps the second model focused on reducing the largest misses instead of perturbing the entire distribution.

## Why GANs Are Not A Great Fit Here
GANs are usually a poor fit when the primary objective is numerical regression error on structured physical data.

The main issue is objective mismatch:
- A GAN tries to make outputs look statistically plausible.
- Your goal is to reduce concrete prediction error in dB.
- Those are not the same objective.

In practice, GAN training often helps appearance or texture, but it does not reliably improve RMSE on tail cases.
For this problem, GANs have several disadvantages:
- They are harder to train and more unstable.
- They can improve average-looking samples while leaving outliers untouched.
- They can even make some rare cases worse while improving visual smoothness.
- They add adversarial dynamics that are unnecessary if the goal is just to correct outliers.

So if the priority is “do not worsen RMSE,” GANs are not the first tool I would reach for.

## Better Second-Stage Design
If the goal is a second model after stage1, these ideas are best used together, not as separate mutually exclusive options:

1. Residual refiner
- Train a second model on the residual left by stage1.
- Predict only a delta or correction.
- Clamp the final output so corrections cannot explode.

2. Uncertainty gate
- Train a lightweight gate to decide whether stage1 is trustworthy.
- Apply a stronger correction only on uncertain or high-risk samples.

3. Tail-focused refiner
- Train on samples from the upper error tail.
- Oversample the hardest NLoS / low-antenna cases.
- Use a robust loss so the model focuses on large misses without chasing every small fluctuation.

4. Mixture-of-experts style second stage
- Keep the main model as the broad predictor.
- Add a second expert that specializes in hard regimes.
- Use a gate to blend them conservatively.

The best interpretation is to combine all three: a residual refiner, controlled by an uncertainty gate, trained preferentially on the upper error tail.
That gives you a conservative correction model that acts only when needed and is explicitly optimized for the worst cases.

## Recommended Training Order
The best order is:

1. Train stage1 to convergence.
2. Freeze stage1 or treat it as a fixed teacher.
3. Train stage2 afterward as a correction model.
4. Evaluate stage2 against stage1 using strict acceptance rules.

That order matters because it gives stage2 a stable target distribution. If stage2 is trained before stage1 has settled, it is learning on a moving target and tends to chase noise.

## What Stage2 Should Optimize
Stage2 should be judged with conservative metrics, not just raw training loss.

The acceptance criteria should include:
- Overall RMSE must not get worse.
- MAE should improve or at least stay close.
- NLoS and low-antenna tail metrics should improve.
- The 90th/95th percentile absolute error should go down.

If stage2 reduces MAE but increases RMSE, that is usually a bad trade for this problem, because it means the tail got worse.

## Practical Interpretation For Try49
For Try49 specifically, the clean story is:
- Stage1 is the main physical regressor.
- The prior confidence channel helps stage1 understand when the formula prior is trustworthy.
- The second model should come later and should combine three roles at once: residual correction, uncertainty gating, and tail-focused training.
- GAN training is not the preferred route.

So yes, it is absolutely possible to build a second model after stage1, and that is the option I would prefer.
I would not train them together unless there is a later need to test a tightly coupled ablation.
The default should be sequential training: big base model first, then a second big model that specializes in the failures left behind by the first.

## Implemented Stage2 Hand-Off
The current Try49 stage2 implementation follows the sequential plan instead of the old GAN path:
- Export stage1 predictions and error maps to HDF5 with `TFGFortyNinthTry49/scripts/export_stage1_outputs_hdf5.py`.
- Train the stage2 tail refiner from those frozen stage1 outputs with `TFGFortyNinthTry49/train_pmnet_tail_refiner.py`.
- Keep the stage2 model separate from stage1 so the first model does not need to remain resident in VRAM during stage2 training.
- Use the dedicated non-GAN stage2 config `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_tail_refiner_fastbatch/fortyninthtry49_pmnet_tail_refiner_stage2.yaml`.
- Launch the flow with the new HDF5 export job and 4-GPU stage2 slurm script in `TFGFortyNinthTry49/cluster/`.

The exported HDF5 files store:
- `stage1_pred_norm_f16`
- `stage1_pred_db_f16`
- `stage1_error_db_f16`
- `stage1_abs_error_db_f16`

The stage2 refiner consumes the stage1 prediction and absolute error maps, then applies residual correction, uncertainty gating, and tail-focused weighting.

## Why Augmentation Is Off
- Try49 stage1 augmentation is disabled so the widened stage1 run produces a stable frozen prior and the exported HDF5 files remain reproducible.
- Try49/Try50 stage2 augmentation is also disabled because the tail refiner appends frozen stage1 maps after the base sample; spatial augmentation would otherwise move the GT/input tensors but leave the stage1 maps out of sync.
- A fixed seed only repeats the same random augmentation choices. It does not keep frozen HDF5 outputs aligned once the stage1 model or export settings change.
- For this sequential chain, deterministic inputs are the safer default while debugging and rerunning the queue.

## Deprecated Try49 YAMLs
- Removed the old top-level Try49 configs after switching to the widened 112-channel stage1 flow:
	- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1.yaml`
	- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage2.yaml`
- Removed the obsolete launcher scripts tied to those old configs:
	- `TFGFortyNinthTry49/cluster/run_fortyninthtry49_pmnet_prior_stage1_baseline_4gpu.slurm`
	- `TFGFortyNinthTry49/cluster/run_fortyninthtry49_pmnet_prior_stage1_priorconf_4gpu.slurm`
	- `TFGFortyNinthTry49/cluster/run_fortyninthtry49_pmnet_prior_stage2_4gpu.slurm`
- The active configs are now the reduced stage1 pair:
	- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_initial.yaml`
	- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_prior_gan_fastbatch/fortyninthtry49_pmnet_prior_stage1_widen112_resume.yaml`
- The new non-GAN stage2 config replaces the old stage2 YAML:
	- `TFGFortyNinthTry49/experiments/fortyninthtry49_pmnet_tail_refiner_fastbatch/fortyninthtry49_pmnet_tail_refiner_stage2.yaml`

## Possible Improvements
- If stage2 still leaves sharp, ripple-like, or otherwise high-frequency residuals, add a small stage3 that specializes in detail correction instead of broad regression.
- A lighter alternative is a frequency-aware auxiliary loss or a Laplacian/high-pass branch on stage2 before committing to a full stage3.
- Keep this as a follow-up only if the stage2 validation maps show that the remaining errors are mostly edge/detail artifacts.
