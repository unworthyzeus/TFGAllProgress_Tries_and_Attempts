# Try 52: Paper-Backed Next Steps

This note answers a practical question:

- what changes are most supported by the literature for path-loss / radio-map
  prediction,
- which of them look relevant for our current bottlenecks,
- and whether reinforcement learning is a good next step.

The current context is:

- `Try 51 stage1` is already closer to the common supervised setup from the
  literature;
- `LoS` is not the main bottleneck anymore;
- `NLoS` remains the dominant source of error;
- the target is still far from the desired overall RMSE.

## Short answer

The next strong, paper-backed jumps are more likely to come from:

1. better supervised pretraining / transfer learning;
2. explicit `LoS/NLoS` routing with specialized regressors;
3. stronger architecture for spatial reasoning, especially transformer-conv
   hybrids or mixture-of-experts style routing;
4. ensembling and uncertainty-aware selection.

Reinforcement learning is **not** the usual method for training the path-loss
 predictor itself.

The RL papers we found mostly use RL for:

- data collection,
- radio-map exploration,
- or UAV trajectory planning,

not for the core supervised map-to-pathloss regression model.

## Why RL is probably not the right next step

RL appears in the radio-map literature mainly when the task is:

- deciding where a robot/UAV should move to collect better measurements, or
- adapting online while exploring an unknown environment.

Examples:

- Clark et al., "PropEM-L: Radio Propagation Environment Modeling and Learning
  for Communication-Aware Multi-Robot Exploration"
  - [arXiv](https://arxiv.org/abs/2205.01267)
- Wang et al., "UAV Path Planning and Radio Mapping Based on Deep
  Reinforcement Learning"
  - [Journal page](https://www.jas.shu.edu.cn/EN/10.3969/j.issn.0255-8297.2024.02.002)

Those papers are valuable, but they solve a different problem:

- they improve *where to sample* or *how to explore*,
- not mainly *how to train the base path-loss predictor from map inputs*.

For our current setup, RL would add complexity before fixing the more standard
supervised bottlenecks.

## What the stronger papers repeatedly do

## 1. Stay in supervised regression

This remains the dominant pattern in the stronger path-loss / radio-map papers:

- RadioUNet:
  - [arXiv](https://arxiv.org/abs/1911.09002)
- PMNet:
  - [arXiv](https://arxiv.org/abs/2211.10527)
- PMNet with transfer learning:
  - [arXiv](https://arxiv.org/abs/2312.03950)

Repeated pattern:

- physical or map prior as input;
- supervised loss;
- validation-driven model selection;
- transfer from one scenario to another if data are scarce.

This strongly supports keeping the core training loop supervised rather than
switching to RL.

## 2. Use transfer learning when moving to new environments

The PMNet transfer-learning extension is especially relevant:

- Lee and Molisch, "A Scalable and Generalizable Pathloss Map Prediction"
  - [arXiv](https://arxiv.org/abs/2312.03950)

What matters:

- transfer learning allows a pretrained model to learn a new scenario
  faster and with less data;
- the paper reports faster adaptation and lower data requirements while
  preserving accuracy.

Why it matters for us:

- if the real requirement is "adapt to new cities,"
- then pretraining on broader simulated or auxiliary scenarios and then
  fine-tuning on the target distribution is more literature-aligned than
  city-specific memorization.

## 3. Split or route `LoS` and `NLoS` explicitly

One of the most directly relevant references is:

- Krecic et al., "A Machine Learning Approach for Path Loss Prediction Using
  Combination of Regression and Classification Models"
  - [Sensors 2024](https://www.mdpi.com/1424-8220/24/17/5855)

What matters:

- the model uses a classifier to estimate whether the sample belongs more to
  `LoS` or `NLoS`;
- it then combines or routes between different regressors;
- both hard and soft routing are considered.

This is very close to our current intuition:

- `LoS` is already much easier;
- `NLoS` behaves like a different regime;
- a single regressor is probably too blunt.

Paper-backed implication for `Try 52`:

- make `LoS/NLoS` routing explicit in the learned system,
- not only as an analysis metric.

## 4. Use more expressive architectures than plain CNNs when spatial context is hard

Recent papers push toward hybrid transformer-convolution models:

- RMTransformer:
  - [arXiv](https://arxiv.org/abs/2501.05190)
- TransPathNet:
  - [arXiv](https://arxiv.org/abs/2501.16023)
- Transformer-based neural surrogate for variable-sized maps:
  - [arXiv](https://arxiv.org/abs/2310.04570)

Common message:

- pure convolutions may miss the longer-range dependencies or irregular
  geometry needed for hard radio-map prediction;
- transformer-based attention can help focus on the spatial regions that
  matter most for attenuation and blockage.

For us this suggests:

- keep a convolutional decoder for dense map output,
- but consider a stronger encoder or attention blocks in the bottleneck,
- especially if `NLoS` depends on larger contextual structures.

## 5. Ensemble methods are still competitive and simple

Another strong but simpler line is ensembling:

- Kwon and Son, "Accurate Path Loss Prediction Using a Neural Network Ensemble
  Method"
  - [Sensors 2024](https://www.mdpi.com/1424-8220/24/1/304)

What matters:

- the paper explicitly improves prediction by combining several networks with
  diverse hyperparameters;
- the final predictor is the integrated output of multiple trained ANNs.

For us, this is attractive because:

- it is low risk relative to redesigning the whole model;
- it can be added after a better supervised branch already exists;
- it may reduce variance and stabilize cross-city generalization.

## 6. Hybrid physics + learning still looks strong

Hybrid approaches continue to appear in newer work:

- FERMI:
  - [arXiv](https://arxiv.org/abs/2504.14862)
- PropEM-L:
  - [arXiv](https://arxiv.org/abs/2205.01267)

The common lesson is:

- use physics for the easy/direct component,
- and learning for the hard environmental interactions.

That remains highly aligned with our original prior-plus-residual idea.
The problem is not the hybrid philosophy itself, but that the learned `NLoS`
correction is still too weak.

## Recommended Try 52 design

If we want a version that is more strongly grounded in the literature than
`Try 49`/`Try 50`, the most defendable next architecture is:

1. `Stage 1`: supervised residual predictor with stronger encoder
   - keep physical prior input;
   - keep supervised regression;
   - replace plain PMNet bottleneck with a transformer-conv hybrid block or
     attention bottleneck;
   - use geography-based holdout.

2. explicit `LoS/NLoS` routing
   - predict `LoS/NLoS` (or consume the mask if it is known and trusted);
   - route to two specialized residual heads:
     - `LoS head`
     - `NLoS head`

3. optional city-type expertization
   - do **not** specialize by city name;
   - if expertization is needed, route by automatically inferred morphology /
     `city_type`, not city identity.

4. `Stage 2`: residual refiner only after stage1 is solid
   - keep it simple;
   - use it to recover remaining local errors,
   - not to carry the whole `NLoS` burden alone.

5. ensemble only after a strong single model exists
   - average a few strong checkpoints / branches;
   - do not use ensemble as a substitute for fixing the weak base model.

## Concrete paper-backed options, ranked

### Highest priority

- transfer learning / pretraining on broader simulated or auxiliary data
  - strongest direct support:
    - [PMNet TL](https://arxiv.org/abs/2312.03950)
- explicit `LoS/NLoS` routing with separate regressors or heads
  - strongest direct support:
    - [classification + regression compound model](https://www.mdpi.com/1424-8220/24/17/5855)
- stronger global-context encoder
  - strongest direct support:
    - [RMTransformer](https://arxiv.org/abs/2501.05190)
    - [Transformer surrogate](https://arxiv.org/abs/2310.04570)

### Medium priority

- small ensemble of top checkpoints / branches
  - support:
    - [neural network ensemble method](https://www.mdpi.com/1424-8220/24/1/304)
- uncertainty-aware routing or confidence output
  - naturally follows from the classifier-plus-expert view in the compound
    model paper.

### Lower priority for now

- reinforcement learning
  - more relevant to sampling / path planning than supervised map prediction:
    - [PropEM-L](https://arxiv.org/abs/2205.01267)
    - [UAV path planning + radio mapping RL](https://www.jas.shu.edu.cn/EN/10.3969/j.issn.0255-8297.2024.02.002)

## Practical conclusion

If the target is still to push overall RMSE much lower, the most paper-backed
next bet is not RL.

The strongest next move is:

- `Try 52 = supervised hybrid model + explicit LoS/NLoS routing + better
  global-context encoder + geography-aware validation`

And if we can support it with broader synthetic / RT pretraining, even better.

That is much closer to where the literature is currently strongest than:

- more adversarial tuning,
- or using RL as the main training paradigm for the predictor itself.
