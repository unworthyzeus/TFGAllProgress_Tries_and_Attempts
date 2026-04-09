# PMNet vs PMHHNet

This note explains the architectural difference between the original
`PMNetResidualRegressor` and the new `PMHHNetResidualRegressor` used in
`Try 54`.

Relevant code:

- [model_pmhhnet.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/model_pmhhnet.py)
- [train_partitioned_pathloss_expert.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/train_partitioned_pathloss_expert.py)
- [generate_try54_configs.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/scripts/generate_try54_configs.py)

## Short version

`PMNet` is a prior-residual regressor with a strong convolutional encoder,
dilated context, and top-down fusion.

`PMHHNet` keeps that base, but adds two things:

1. an explicit **high-frequency branch**
2. a continuous **height-conditioning path**

In `Try 54` we also pair `PMHHNet` with a small auxiliary `no_data` head so the
network can learn where path-loss supervision is missing instead of only
dropping those pixels from the total loss.

The goal is to make one model per topology class generalize across all antenna
heights instead of splitting the experts by height.

## Why PMNet was not enough for Try 54

`PMNet` is good at:

- learning a residual over a calibrated prior;
- combining local features with broader context;
- and staying fairly compact compared with a very large monolithic model.

But in the new `Try 54` setup we want something more specific:

- only `6` experts, not `18`;
- one expert per topology class;
- and **all antenna heights inside the same expert**.

That means the network has to do more than just read a broadcast scalar
channel. It has to **change its internal behavior continuously** as antenna
height changes.

That is what `PMHHNet` is for.

## PMNet

Main class:

- `PMNetResidualRegressor`

Main structure:

1. stem
2. 4 encoder stages
3. dilated context block
4. top-down fusion
5. residual head

Output:

- one residual map
- final prediction is:
  - `prior + residual`

What it does well:

- medium-size path-loss regression
- strong prior + residual learning
- spatial context without becoming huge

What it does not do explicitly:

- no dedicated high-frequency pathway
- no explicit continuous modulation by antenna height inside the feature
  hierarchy

## PMHHNet

Main class:

- `PMHHNetResidualRegressor`

`PMHHNet` is:

- `PMNet`
- plus the high-frequency idea from the earlier `PMHNet`
- plus a height-conditioning pathway

So the logic is:

1. keep the PMNet residual backbone
2. add a lightweight high-pass branch for sharper local structure
3. add FiLM-style modulation from the antenna-height scalar

## The high-frequency branch

`PMHHNet` computes a fixed Laplacian high-pass map from the input tensor and
then learns a small projection branch from that signal.

Why:

- blockers, shadow boundaries, and abrupt local topology changes are partly a
  high-frequency problem
- a plain residual CNN can learn this implicitly, but it is easy for that
  detail to get washed out
- the explicit branch makes those details easier to preserve and reuse

This is especially useful in path-loss prediction because:

- small changes in local obstruction geometry can strongly affect the residual
  over the prior

## The height-conditioning path

This is the key difference.

`PMNet` can receive the antenna height as a constant spatial channel, but that
is still a weak form of conditioning: the same scalar is just repeated across
the whole image.

`PMHHNet` instead treats height as a **continuous control signal**.

Current mechanism:

1. the normalized antenna height is encoded with a small MLP
2. that embedding drives FiLM-style affine modulation
3. the modulation is applied at several places:
   - stem features
   - deep context features
   - high-frequency branch
   - fused features before the head

So instead of saying:

- "here is height as another image channel"

the model can do something closer to:

- "if height is low, amplify features related to local blockers"
- "if height is high, rely relatively more on broader context"

That is a much better fit for generalizing height continuously.

## Architectural difference in one sentence

`PMNet` learns:

- `residual = f(x)`

`PMHHNet` learns:

- `residual = f(x, height)`

but not only by concatenation. It uses:

- explicit high-frequency extraction
- and internal feature modulation driven by height

## Why this matters for Try 54

`Try 54` used to be headed toward:

- `6 topology classes x 3 antenna bins = 18 experts`

The new idea is cleaner:

- `6 experts total`
- one per topology class
- all heights inside each expert

That only makes sense if the model can genuinely adapt to height.

`PMHHNet` is the mechanism that makes that plausible.

Without it, one of two things tends to happen:

1. the expert underfits the height variation
2. or we split again by height and lose the benefit of the simpler expert grid

## Comparison table

| Aspect | PMNet | PMHHNet |
|---|---|---|
| Backbone | PMNet residual encoder + context + top-down fusion | Same base |
| Prior use | Residual over calibrated prior | Residual over calibrated prior |
| High-frequency path | No explicit branch | Yes, Laplacian-driven learned branch |
| Height handling | Usually scalar channel only | Continuous FiLM-style conditioning |
| Missing-supervision handling in Try 54 | Usually masked out only | Residual map + auxiliary `no_data` head |
| Intended regime | Generic prior-residual path-loss regression | Topology specialist that generalizes across antenna heights |
| Try 54 role | Baseline reference | Main expert architecture |

## Compute trade-off

`PMHHNet` is a bit more expensive than plain `PMNet`, because it adds:

- one scalar-conditioning MLP
- several small affine modulators
- one extra high-frequency projection branch

But it is still meant to stay **small enough** for the specialist setting.

That is why `Try 54` uses:

- `6` specialists
- moderate widths
- and one continuous-height expert per topology

instead of one huge model for everything.

## Mental model

You can think of the two models like this:

- `PMNet`: a solid medium path-loss residual network
- `PMHHNet`: PMNet with a sharper eye for local geometry and a better internal
  understanding of antenna height

## Current Try 54 choice

The active `Try 54` direction is:

- topology-routed specialists
- `PMHHNet` for the experts
- height as a continuous scalar condition
- no height-specific expert split

That is the main reason the branch moved from:

- "one model per topology and height"

to:

- "one model per topology, all heights together"
