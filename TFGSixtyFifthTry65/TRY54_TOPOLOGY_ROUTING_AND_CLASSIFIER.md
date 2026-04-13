# Try 54 Topology Routing And Automatic Classifier

This note explains:

- how `Try 54` assigns each sample to one of the `6` topology experts;
- why that routing exists;
- what counts as ground truth vs pseudo-ground-truth;
- and how the future automatic classifier should work on unseen data.

Relevant files:

- [data_utils.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/data_utils.py)
- [model_topology_classifier.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/model_topology_classifier.py)
- [train_topology_classifier.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/train_topology_classifier.py)
- [generate_try54_configs.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/scripts/generate_try54_configs.py)
- [fiftyfourthtry54_topology_classifier.yaml](C:/TFG/TFGpractice/TFGFiftyFourthTry54/experiments/fiftyfourthtry54_topology_classifier/fiftyfourthtry54_topology_classifier.yaml)
- [fiftyfourthtry54_expert_registry.yaml](C:/TFG/TFGpractice/TFGFiftyFourthTry54/experiments/fiftyfourthtry54_partitioned_stage1/fiftyfourthtry54_expert_registry.yaml)

## Why we are not routing by city name

`Try 54` is designed to generalize to new cities.

That means the routing logic should depend on:

- what the built environment looks like;
- how dense the buildings are;
- how tall the positive-height structures are;

and **not** on a memorized city identifier.

This is why `Try 54` routes by morphology derived from `topology_map`, not by
`city`.

The intent is:

- experts specialize in propagation regimes that are visually and physically
  similar;
- the classifier later learns to reproduce that morphology-based routing;
- unseen cities can still be assigned to a reasonable expert family.

## What is true GT and what is pseudo-GT

There are two different supervision layers in `Try 54`.

### Physical GT

This is the real regression target:

- dataset field: `path_loss`
- used to train each PMHHNet expert

This is the actual path-loss supervision and is the thing we care about in the
final metrics.

### Routing pseudo-GT

This is the label that tells us which expert should see a sample during
training.

It is **not** predicted by another model first.
It is computed directly from the sample's `topology_map`.

So:

- experts are trained before the classifier;
- the classifier is trained later to imitate this same deterministic routing;
- routing pseudo-GT is derived from the sample itself, not guessed by a model.

## How the topology class is computed today

The current logic lives in:

- `_compute_try54_partition_metadata(...)`
- `_infer_try54_topology_class(...)`

inside [data_utils.py](C:/TFG/TFGpractice/TFGFiftyFourthTry54/data_utils.py).

For each sample we compute:

- `building_density`
  - fraction of pixels whose topology height is different from the configured
    `non_ground_threshold`
- `mean_height`
  - mean positive building height over the non-ground pixels

The current thresholds are:

- `density_q1 = 0.12`
- `density_q2 = 0.28`
- `height_q1 = 12.0`
- `height_q2 = 28.0`

These thresholds define the `6` classes:

1. `open_sparse_lowrise`
   - low density
   - low mean height

2. `open_sparse_vertical`
   - low density
   - but taller structures

3. `mixed_compact_lowrise`
   - medium density
   - low mean height

4. `mixed_compact_midrise`
   - medium density
   - higher mean height

5. `dense_block_midrise`
   - high density
   - not yet in the tallest bucket

6. `dense_block_highrise`
   - high density
   - high mean height

This gives us a deterministic morphology label for every sample.

## Why this partitioning makes sense

The idea is not that these `6` labels are perfect urban-planning categories.
The idea is that they produce useful expert partitions for propagation.

Why this is reasonable:

- sparse lowrise scenes often behave differently from dense block scenes;
- highrise obstruction patterns change diffraction, blockage, and fine spatial
  transitions;
- one small expert is more likely to learn a coherent regime than one global
  model forced to fit everything at once;
- the classifier can later learn these routing regions because they are derived
  from topology, which it will also see at inference time.

This also matches the broader intuition from the papers we reviewed:

- use morphology/regime structure when the physics differ enough;
- avoid depending on city IDs if the goal is cross-city generalization.

## How experts are trained before the classifier exists

This is the full training workflow right now:

1. read the sample's `topology_map`
2. compute `building_density` and `mean_height`
3. infer the deterministic topology class
4. assign that sample to exactly one of the `6` expert datasets
5. train the expert on real `path_loss` GT

So the classifier is not needed yet.

This is why the expert training data is still "correct":

- routing comes from deterministic sample metadata;
- path-loss supervision comes from the true dataset target.

## Current split sizes for the 6 experts

These are the current counts for the active `Try 54` expert YAMLs:

- split mode: `city_holdout`
- image size: `513 x 513`
- thresholds:
  - `density_q1 = 0.12`
  - `density_q2 = 0.28`
  - `height_q1 = 12.0`
  - `height_q2 = 28.0`

Current sample counts:

| Topology class | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| `open_sparse_lowrise` | 1560 | 400 | 350 | 2310 |
| `open_sparse_vertical` | 1070 | 230 | 230 | 1530 |
| `mixed_compact_lowrise` | 1705 | 475 | 505 | 2685 |
| `mixed_compact_midrise` | 2095 | 965 | 565 | 3625 |
| `dense_block_midrise` | 3565 | 925 | 795 | 5285 |
| `dense_block_highrise` | 405 | 225 | 115 | 745 |

Important note:

- these counts are not arbitrary hand-picked partitions;
- they are the result of:
  - the deterministic topology heuristic;
  - the current `city_holdout` split;
  - and the current threshold values.

So if we later change:

- topology thresholds,
- split seed,
- or split policy,

the counts can move as well.

## How the automatic classifier should work on new data

The classifier's job is simple:

- input: the same topology information available at inference time
- output: one of the `6` topology classes

The target for classifier training is the deterministic label produced by the
heuristic above.

### Training stage

During classifier training:

- each sample gets a topology class from the heuristic;
- that class becomes the classifier target;
- the classifier learns to approximate the same routing function directly from
  the input map.

### Inference stage

For unseen data:

1. compute or load the topology map
2. run the topology classifier
3. obtain one of the `6` classes
4. look up the matching expert in the registry
5. run the corresponding PMHHNet expert

So the final runtime path should be:

```text
topology map -> topology classifier -> topology class -> expert registry -> PMHHNet expert
```

## Why the classifier is still useful if the heuristic already exists

There are two reasons:

1. operational simplicity
   - at deployment time it is often cleaner to use one learned router than to
     re-implement a threshold pipeline in every environment

2. learnable robustness
   - the classifier can learn smoother boundaries than a hard threshold
   - if the heuristic is slightly noisy, the classifier may still produce more
     stable routing decisions from real topology inputs

The heuristic is still important because:

- it gives us deterministic labels;
- and it keeps the routing policy interpretable.

## Relationship with antenna height

In `Try 54`, antenna height is **not** part of the expert routing key anymore.

That is deliberate.

The design is:

- route by topology only;
- let each expert learn all heights;
- condition the expert continuously on antenna height through PMHHNet's scalar
  FiLM path.

This should generalize better than exploding the expert grid into:

- `6 topology classes x 3 height bins`

which would give many smaller, more fragmented subsets.

## What we should improve later

The current routing heuristic is intentionally simple and interpretable.
Future improvements that still preserve the same philosophy:

- recalibrate the density and height thresholds from dataset statistics instead
  of keeping them fixed by hand;
- add one or two extra morphology features if needed:
  - connected-component size
  - edge density
  - local obstruction ratio
- measure classifier confusion against the heuristic labels on held-out cities;
- compare hard routing vs top-2 routing if one expert family becomes too broad.

For now, the important point is:

- experts are trained on real path-loss GT;
- routing labels come from deterministic topology-derived pseudo-GT;
- and the automatic classifier is meant to reproduce that routing for unseen
  cities, not replace the path-loss supervision itself.
