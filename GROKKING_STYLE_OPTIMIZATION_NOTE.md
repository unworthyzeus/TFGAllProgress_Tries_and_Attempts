# Try 55/56 Training-Dynamics Note

This note documents a recurrent failure mode we observed in `Try 55`, and the
optimizer changes we decided to test in both `Try 55` and `Try 56`.

## What we observed

In several experts, the training curves showed the same pattern:

- `train RMSE` kept going down;
- `generator loss` kept going down;
- but `val RMSE` stopped improving after an early best epoch and then drifted up.

For example, in `dense_block_highrise`, the validation RMSE reached its best
around epoch `9`, while training RMSE continued improving for many more epochs.
That is a classic sign that optimization is still fitting the training set but
is no longer moving toward better validation behavior.

## Our working hypothesis

We are not claiming that these path-loss experts will literally exhibit
textbook grokking. This is a regression problem, not modular arithmetic or
algorithmic classification. Still, the training dynamics looked similar enough
to justify borrowing part of the grokking optimization toolbox:

- avoid dropping the learning rate too early;
- keep enough optimization "energy" late in training;
- use stronger weight decay so the optimizer is pushed away from purely
  memorizing solutions.

There is also an engineering hypothesis specific to this dataset:

- the stored supervision is quantized (`uint8`), so the target landscape is
  relatively coarse;
- a higher constant learning rate may help the model move between coarse target
  bins instead of settling too early into a shallow memorizing basin.

That last point is an inference from our setup, not a direct claim from the
papers below.

## What the literature says

### 1. Weight decay can make the generalization transition happen earlier

The ICLR 2024 paper *Grokking with Large Initialization and Small Weight Decay*
reports that increasing weight decay makes the grokking transition happen
earlier, but also warns that too much weight decay can eventually hurt
optimization and collapse training.

Source:
- [ICLR 2024 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/909c8fef63e1cede406ce9e6794f99a2-Paper-Conference.pdf)

Relevant points from the paper:
- increasing weight decay can make the transition happen earlier;
- too much weight decay can become so strong that it hurts training;
- with zero weight decay, generalization can still appear, but the transition is
  much less sharp and much slower.

### 2. Constant high learning-rate pressure can help escape the slow regime

The same ICLR 2024 paper also discusses the no-weight-decay setting, where
generalization can still emerge but much more slowly, and where maintaining the
right optimization dynamics matters.

Source:
- [ICLR 2024 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/909c8fef63e1cede406ce9e6794f99a2-Paper-Conference.pdf)

### 3. Accelerating late generalization is a real optimization theme

`Grokfast` is useful as motivation here: its entire premise is that late
generalization can sometimes be accelerated by changing optimization dynamics
rather than only changing the model architecture.

Source:
- [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233)

We are not implementing Grokfast itself here, but it supports the broader idea
that optimizer dynamics matter a lot once the model has entered the
"train improves, validation stalls" regime.

## What we changed

### Try 55

We switched the expert configs to:

- `optimizer = adamw`
- `learning_rate = 1.0e-3`
- `weight_decay = 0.25`
- `lr_scheduler = none`

### Try 56

We applied the same philosophy, but a bit more conservatively because `Try 56`
predicts `delay_spread` and `angular_spread` rather than path loss:

- `generator_optimizer = adamw`
- `generator_lr = 3.0e-4`
- `weight_decay = 0.25`
- `lr_scheduler = none`

## Why AdamW

Once weight decay becomes very large, we prefer to make it explicit and
decoupled instead of mixing it implicitly into the gradient as in classic Adam.
That makes the experiment easier to reason about.

## What to monitor next

These changes are deliberately unusual, so they should be judged by curves, not
intuition.

We should watch for:

- whether `val RMSE` stops drifting upward after the early best epoch;
- whether the best epoch moves later in training;
- whether training collapses, which would indicate that `weight_decay = 0.25`
  is too aggressive for a given expert.

If training starts collapsing, the first rollback should be:

- lower `weight_decay` from `0.25` to `0.10` or `0.05`

rather than immediately lowering the learning rate.
