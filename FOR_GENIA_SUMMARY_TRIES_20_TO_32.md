# Supervisor Summary of Tries 20-32

This note is written for supervisor-facing discussion. It summarizes what was tested from `Try 20` onward, what each idea means in practical terms, and what has or has not improved according to the recent plots and validation metrics.

The main point is that these tries were not random hyperparameter changes. Each one was opened to test a specific hypothesis after inspecting the prediction maps and the training curves.

## 1. Starting point

Before `Try 20`, the project already had:

- a stable HDF5 training pipeline,
- separate path-loss and spread branches,
- antenna-height conditioning,
- FiLM-based scalar conditioning,
- and a stronger modern baseline from `Try 15-19`.

So the recent stage was no longer about "making the pipeline work". It was about diagnosing which bottlenecks were still limiting prediction quality.

## 2. Main visual and metric problems that triggered the newer tries

By directly inspecting exported predictions, several recurring issues appeared:

- some `path_loss` predictions showed decoder artifacts or overly smooth large-scale fields;
- the transmitter-centered radial carrier pattern in `path_loss` was still underlearned;
- `delay_spread` and `angular_spread` were often too faint, too smooth, or too contrast-compressed;
- strong sparse responses in the spread maps were frequently underestimated;
- adding complexity alone did not consistently improve the strongest baseline.

These observations motivated the sequence of tries from `20` onward.

## 3. What each try meant

### Try 20: bilinear decoder

`Try 20` replaced transposed convolutions with bilinear upsampling plus standard convolution.

What it means:

- instead of learning the upsampling directly with deconvolution, the model first upsamples more smoothly and then refines with a convolution;
- this is often used to reduce checkerboard-like artifacts.

Why it was tested:

- some `path_loss` predictions showed grid-like or decoder-related artifacts.

What it taught:

- it was a reasonable cleanup step, but by itself it did not solve the remaining large-scale physical errors.

### Try 21: multiscale loss

`Try 21` added multiscale supervision for `path_loss`.

What it means:

- the prediction is compared with the target not only at full resolution, but also at coarser resolutions;
- this gives the model pressure to match both local detail and global field structure.

Why it was tested:

- some errors in `path_loss` looked global, not just local.

What it taught:

- multiscale supervision was useful, but the best behavior appeared when it was combined with the decoder fix rather than used alone.

### Try 22: bilinear decoder + multiscale path-loss loss

`Try 22` combined the two previous ideas and became the strongest clean path-loss baseline of this family.

What it means:

- a cleaner decoder,
- plus supervision that explicitly checks the field at several spatial scales.

What worked:

- this became the best recent `path_loss` branch;
- it established the baseline that later path-loss tries had to beat.

What did not solve the whole problem:

- even this model still underlearned the radial propagation pattern around the transmitter.

### Try 23: spread branch with the Try 22 recipe

`Try 23` transferred the `Try 22` philosophy to `delay_spread` and `angular_spread`.

What it means:

- bilinear decoder for spread prediction,
- multiscale regression loss generalized from path loss to continuous spread targets.

Why it was tested:

- the spread outputs also looked too smooth and structurally weak.

What it taught:

- the spread branch did benefit from better structure-aware supervision,
- but the problem was not only structural; amplitudes were still being underestimated.

### Try 24: prepared multitask branch

`Try 24` was prepared as a multitask branch for:

- `path_loss`
- `delay_spread`
- `angular_spread`

It was intentionally kept local and not used as the main active comparison.

What it means:

- one shared model would predict all targets at once.

Why it was not the immediate priority:

- the single-task branches were more interpretable while the project was still diagnosing bottlenecks.

### Try 25: lightweight bottleneck attention

`Try 25` added attention in the bottleneck of the path-loss network.

What attention means here:

- the model tries to mix information from distant spatial locations more explicitly,
- instead of relying only on convolutional receptive fields.

Why it was tested:

- after `Try 22`, a plausible remaining hypothesis was that the model still lacked long-range context.

What it taught:

- attention could look promising early,
- but it did not clearly surpass the simpler `Try 22` baseline.

### Try 26: gradient-aware spread loss

`Try 26` extended the spread branch with a gradient-aware loss.

What that means:

- the loss does not only compare values;
- it also compares spatial changes, so sharp transitions matter more.

Why it was tested:

- the spread maps often looked too blob-like and too flat.

What worked:

- this became the strongest recent spread baseline;
- compared with `Try 23`, it preserved spread structure better.

### Try 27: topology-edge-weighted path-loss regularization

`Try 27` weighted path-loss errors more strongly near topology edges.

What it means:

- errors close to obstacle transitions or sharp topology changes count more in the loss.

Why it was tested:

- some visible path-loss errors seemed concentrated near physically delicate urban transitions.

What it taught:

- topology-aware weighting was a reasonable idea,
- but it did not remove the dominant remaining path-loss bottleneck.

### Try 28: attention + topology-edge weighting

`Try 28` combined `Try 25` and `Try 27`.

What it means:

- more global context through attention,
- plus more local pressure near topology edges.

Why it was tested:

- the two ideas targeted different possible failure modes and could have been complementary.

What happened:

- it stayed competitive,
- but it still did not clearly beat `Try 22`.

What that taught:

- the main missing ingredient in `path_loss` was probably not just context or topology-edge emphasis.

### Try 29: radial profile loss + radial gradient loss

`Try 29` was opened after manually reviewing 20 composite diagnostic panels.

What the review showed:

- the main remaining path-loss problem was the weak reconstruction of the transmitter-centered radial pattern.

What the new losses mean:

- a radial profile loss checks whether average values over concentric rings match the target;
- a radial gradient loss checks whether ring-to-ring changes also match the target.

Why it was tested:

- the visual evidence suggested that the model was learning some urban modulation, but not the underlying propagation carrier strongly enough.

What happened:

- this was physically better motivated than simply adding complexity,
- but it did not surpass `Try 22` in practice.

### Try 30: value-weighted + hotspot-focused spread losses

`Try 30` started from `Try 26` and added two spread-specific losses.

What they mean:

- value-weighted loss gives more importance to large target values, so strong spread responses are not overwhelmed by the dark background;
- hotspot-focused loss adds extra pressure on the strongest regions of the target map.

Why it was tested:

- the visual review showed that spread predictions often got the approximate support right but attenuated the peaks too much.

What happened:

- the idea was well motivated,
- but it did not improve over `Try 26`.

### Try 31: physical prior + learned residual for path loss

`Try 31` was the first real paradigm shift for `path_loss`.

physical prior + learned residual (also inherits try 22 but not try 26 or try 29):

- instead of predicting the whole path-loss map from scratch, the model starts from a simple physical prior based on distance and carrier frequency (7.125GHz);
(Not a good formula $PL_{prior} (dB)=20log_{10} (d_{3D} )+20log_{10} (fGHz)+32.45$) (frequency is 7.125GHz)
- the network then learns only the residual correction on top of that prior.

In simple terms:

- prior = "what a simple propagation law would predict";
- learned residual = "how the real environment deviates from that simple law".

Why it was tested:

- the earlier tries suggested that the network was struggling to reconstruct the base radial propagation field.

What happened so far:

- conceptually this is the right type of idea for the main bottleneck,
- but in the current form it is still not outperforming `Try 22`.

### Try 32: support + amplitude prediction for spreads

`Try 32` was the equivalent paradigm shift for `delay_spread` and `angular_spread`.

support + amplitude

- the model no longer predicts only one map per target;
- it predicts:
  - where the response should exist at all, called the support,
  - and how strong it should be there, called the amplitude.

In simple terms:

- support answers "where is the active region?";
- amplitude answers "how large is the value inside that region?".

The final prediction is built by combining both.

Why it was tested:

- the earlier spread tries suggested that the model often knew the rough location of activity but underestimated the magnitude.

What happened so far:

- the idea is principled,
- but the current implementation is not yet beating `Try 26`.

## 4. What has worked best so far

### Path loss

The strongest recent result is still `Try 22`.

What seems to work for path loss:

- bilinear decoder,
- multiscale path-loss supervision,
- keeping the branch clean and focused.

What has not clearly helped enough:

- adding attention alone,
- adding topology-edge weighting alone,
- combining both,
- radial auxiliary losses in their current form,
- and the first prior-plus-residual formulation in `Try 31`.

### Delay spread and angular spread

The strongest recent spread result is still `Try 26`.

What seems to work for spreads:

- bilinear decoder,
- multiscale regression supervision,
- gradient-aware structural pressure.

What has not clearly helped enough:

- more aggressive weighting toward peaks in `Try 30`,
- or the support-plus-amplitude reformulation of `Try 32` in its current form.

## 5. Current interpretation

The recent experiments suggest two different conclusions:

- for `path_loss`, the project still needs a better way to inject the base propagation law without overconstraining the model;
- for `delay_spread` and `angular_spread`, the project still needs a better way to protect sparse strong responses without making optimization unstable or biased.

So the main lesson from `Try 20-32` is not that the new ideas were useless. The real lesson is more precise:

- the simpler structural fixes were genuinely helpful;
- later, more ambitious ideas were informative even when they did not win, because they revealed which hypotheses were not the real bottleneck.

## 6. Short supervisor-facing conclusion

If this has to be reduced to a short summary:

- `Try 20-22` established the strongest modern `path_loss` baseline by fixing the decoder and adding multiscale supervision.
- `Try 23` and especially `Try 26` established the strongest modern spread baseline by transferring those structural ideas and then adding gradient-aware supervision.
- `Try 25`, `Try 27`, and `Try 28` tested whether extra context or topology-aware weighting would close the remaining path-loss gap, but they did not clearly beat `Try 22`.
- After manual review of 20 diagnostic composite panels, `Try 29` and `Try 30` were opened to target the newly identified failure modes directly, but they still did not improve over the best baselines.
- `Try 31` and `Try 32` represent a true change of formulation rather than a small tweak:
  - `Try 31` predicts a learned residual on top of a physical prior for `path_loss`;
  - `Try 32` predicts support and amplitude separately for the spread targets.
- At the moment, these paradigm-shift ideas are promising conceptually, but the strongest validated baselines are still `Try 22` for `path_loss` and `Try 26` for `delay_spread` and `angular_spread`.
