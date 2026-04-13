# Try 57/56 Overfitting Mitigation Options

This note collects the main options we should consider **apart from the
grokking-style optimizer hypothesis**.

The recurrent pattern in our experts has been:

- `train RMSE` keeps improving;
- `val RMSE` peaks early and then gets worse;
- the gap is strongest in the smallest or hardest topology partitions.

That is the standard overfitting picture for a city-holdout setup, even when
the optimizer is still making progress on the training set.

## 1. Stronger Early Stopping

This is the safest baseline.

What to do:

- keep selecting by `val RMSE`;
- stop training after `N` epochs without improvement;
- prefer restoring the best checkpoint rather than the last checkpoint.

Why it helps:

- in our curves, the best epoch often arrives much earlier than the end of the
  run;
- once the model starts memorizing topology-specific details from the training
  cities, continuing longer usually hurts validation.

When it is most useful:

- experts with small splits such as `dense_block_highrise`;
- any run where the best epoch is clearly early and the validation curve drifts
  upward afterward.

## 2. More Physics-Safe Augmentation

Augmentation is one of the best standard tools if it does not break the
problem's geometry.

What to consider:

- horizontal and vertical flips only if antenna/prior alignment stays valid;
- `90` degree rotations only if the spatial semantics of the inputs remain
  correct;
- light random masking or corruption on auxiliary channels;
- small noise on continuous conditioning scalars when physically sensible.

Why it helps:

- the expert is forced to learn more stable morphology patterns instead of
  memorizing city-specific layouts.

Warning:

- for radio-map tasks, not every image augmentation is valid;
- we should keep only augmentations that preserve the physical interpretation
  of the sample.

## 3. Reduce Capacity on Small Partitions

If a topology class has few samples, a relatively expressive model can overfit
very quickly.

What to do:

- lower `base_channels`;
- reduce the deepest blocks first;
- keep dropout moderate but present;
- avoid adding extra branches unless they clearly improve validation.

Why it helps:

- smaller experts are less able to memorize idiosyncratic city structure;
- this is especially relevant for `dense_block_highrise`, where the split is
  much smaller than the others.

Practical rule:

- the smaller the partition, the stronger the argument for a lighter model.

## 4. Merge the Weakest Topology Buckets

The `6`-expert routing is attractive, but it can fragment the data too much.

What to consider:

- merge the smallest topology classes into a broader shared expert;
- keep the biggest classes separated;
- compare `6` experts versus `4` experts, not only `6` versus `1`.

Why it helps:

- it trades specialization for better sample efficiency;
- when validation is city-holdout, more training diversity inside each expert
  can matter more than narrow specialization.

This is one of the strongest non-optimizer levers if the smallest experts keep
overfitting.

## 5. Keep the Validation Metric Aligned With the Real Goal

We already improved this in `Try 57`, but the general principle still matters.

What to do:

- validate in the same units we care about (`physical RMSE`);
- monitor both train and validation RMSE in the same unit family;
- keep a separate eye on important slices such as `LoS` and `NLoS`.

Why it helps:

- some runs look better in loss space than they really are;
- if a model improves train loss but not the real validation metric, that is
  overfitting even if optimization looks healthy.

## 6. Preserve the Good Prior Where It Is Already Strong

This matters more for `Try 57` than for `Try 56`, but the idea is useful:

- do not force the network to relearn easy `LoS` structure if the prior already
  handles it well;
- bias learning capacity toward the harder regimes.

Why it helps:

- overfitting often starts by making unnecessary corrections in easy regions;
- if the model spends capacity perturbing already-good areas, validation gets
  worse without much train penalty.

Possible implementation ideas:

- smaller correction scale in `LoS`;
- explicit `LoS` preservation term;
- regime-aware monitoring for checkpoint selection.

## 7. Exponential Moving Average of Weights

EMA is a practical way to reduce the noise of late training.

What to do:

- keep a moving average of model weights during training;
- validate with the EMA weights, not only the raw last-step weights.

Why it helps:

- late epochs can wander even if the overall optimization trend is good;
- EMA often improves validation smoothness and checkpoint quality with little
  conceptual complexity.

This is a strong candidate if we want a low-risk upgrade before redesigning the
architecture again.

More detail:

- EMA keeps a second copy of the model weights alongside the normal training
  weights;
- after each optimizer step, that second copy is updated with:

  `ema_weights = decay * ema_weights + (1 - decay) * current_weights`

- the normal model still learns exactly as before;
- the EMA model is only a smoothed version of the training trajectory.

Intuition:

- the raw model represents what the optimizer learned most recently;
- the EMA model represents what the optimizer has learned consistently over a
  longer window.

Why this can help our runs:

- in `Try 57`, we often see training keep improving while validation becomes
  noisy or drifts upward after the best epoch;
- some of that is true overfitting;
- but some of it can also be late-step instability, especially with small
  experts and noisy minibatch updates;
- EMA reduces that instability without changing the architecture or the loss.

What it does **not** do:

- it does not magically fix a fundamentally overfit model;
- it does not replace early stopping, better augmentation, or expert merging;
- it is best seen as a stabilizer for checkpoint selection and validation.

Practical default for us:

- start with `ema_decay = 0.99`;
- validate and select checkpoints with EMA weights;
- optionally keep saving both the raw model and the EMA model if we want to
  compare them explicitly.

Expected behavior if it helps:

- validation curves become smoother;
- the best epoch is less noisy;
- the selected checkpoint is often a bit better than the raw last-step model.

## 8. Better Expert-Specific Training Budgets

Not every expert should get the same time budget.

What to do:

- allow smaller experts to stop earlier;
- give larger experts more epochs only when validation still improves;
- avoid uniform long runs for all topology classes.

Why it helps:

- small experts tend to reach their best epoch much earlier;
- fixed long training schedules overfit them unnecessarily.

## 9. Rebalance Sampling or the Loss Across Regimes

Even inside one expert, the useful error signal can be highly imbalanced.

What to consider:

- upweight the harder valid regions;
- oversample harder examples;
- reduce the dominance of easy pixels that quickly become memorized.

Why it helps:

- if most pixels are easy, the model can keep improving train RMSE while doing
  little for the genuinely hard regions;
- that is one route to apparent progress with poor validation gains.

## 10. Prefer Simpler Fixes Before New Architecture

Before inventing a new branch or head, the order of operations should usually
be:

1. early stopping;
2. better augmentation;
3. capacity reduction or expert merging;
4. EMA;
5. only then larger architectural changes.

Why:

- these are cheaper to test;
- they isolate the cause of overfitting more cleanly.

## Recommended Priority Order For Us

If we keep seeing the same pattern, the most sensible order is:

1. add stricter early stopping;
2. add EMA evaluation;
3. reduce capacity or merge the smallest expert buckets;
4. enable only physics-safe augmentation;
5. revisit routing granularity if the smallest experts remain unstable.

## Relation To The Grokking-Style Note

The grokking-style optimizer experiment is about changing **optimization
dynamics**:

- constant LR,
- higher LR,
- stronger weight decay.

This note is about the more standard anti-overfitting toolbox:

- stop earlier,
- regularize better,
- reduce fragmentation,
- and make validation/selection more robust.

We should treat them as complementary, not as mutually exclusive ideas.
