# Documentation Index

This index is meant to be a practical navigation map, not just a list of filenames.

The repository now contains many historical notes, branch-specific summaries, reevaluation reports, and source mappings. The goal of this file is to say:

- which documents are the current authoritative ones,
- which ones are best for a supervisor,
- which ones are best for technical implementation,
- and which ones exist mainly for historical traceability.

## Recommended reading order

If someone new needs to understand the current state quickly, the recommended order is:

1. [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)
2. [VERSIONS.md](/C:/TFG/TFGpractice/VERSIONS.md)
3. [TRY14_TRY22_REEVALUATION_AND_MASKING.md](/C:/TFG/TFGpractice/TRY14_TRY22_REEVALUATION_AND_MASKING.md)
4. [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)
5. the active-branch note for the current path-loss line
6. [PATH_LOSS_MODEL_TRAINING_PAPERS.md](/C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)
7. [TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md](/C:/TFG/TFGpractice/TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md)
8. [TRY52_PAPER_BACKED_NEXT_STEPS.md](/C:/TFG/TFGpractice/TRY52_PAPER_BACKED_NEXT_STEPS.md)

Right now the most relevant active-branch note is:

- [TRY47_UNET22_PRIOR_NLOS_MOE.md](/C:/TFG/TFGpractice/TRY47_UNET22_PRIOR_NLOS_MOE.md)
- [TRY49_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY49_PRIOR_IMPROVEMENTS.md)
- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)
- [TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md](/C:/TFG/TFGpractice/TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md)

## Which documents are authoritative right now

The following files should be treated as the main current references.

### Master summaries

- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)
  - long-form merged narrative
  - combines supervisor summary, source motivation, current experimental logic, and interpretation

- [VERSIONS.md](/C:/TFG/TFGpractice/VERSIONS.md)
  - chronological experiment history
  - best for checking what each try was
  - best for fast experiment lookup

### Current path-loss methodology

- [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)
  - how the calibrated prior system works
  - why the JSON is part of the experiment
  - why calibration must be train-only

- [TRY47_UNET22_PRIOR_NLOS_MOE.md](/C:/TFG/TFGpractice/TRY47_UNET22_PRIOR_NLOS_MOE.md)
  - strongest prior-calibration baseline still in practical use
  - source of the copied calibration reused by later branches

- [TRY49_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY49_PRIOR_IMPROVEMENTS.md)
  - current two-stage PMNet branch
  - documents the `mae_dominant` stage1 branch and the `84`-channel stage2

- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)
  - current `Try 50` prior-research status
  - explains that the copied `Try 47` calibration remains the only practical prior

- [TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md](/C:/TFG/TFGpractice/TRY51_LITERATURE_ALIGNED_SUPERVISED_PLAN.md)
  - current plan for the new `Try 51` branch
  - explains why it pivots toward automatic city-type generalization and supervised regime-aware training

- [TRY52_PAPER_BACKED_NEXT_STEPS.md](/C:/TFG/TFGpractice/TRY52_PAPER_BACKED_NEXT_STEPS.md)
  - focused note on the next paper-backed jump after `Try 51`
  - explains why RL is probably not the right next training paradigm
  - proposes a `Try 52` centered on explicit `LoS/NLoS` routing and stronger global context

- [PATH_LOSS_MODEL_TRAINING_PAPERS.md](/C:/TFG/TFGpractice/PATH_LOSS_MODEL_TRAINING_PAPERS.md)
  - literature-backed note on how path-loss and radio-map models are usually trained
  - best short reference when the question is methodological rather than branch-specific

### Current historical correction about metrics

- [TRY14_TRY22_REEVALUATION_AND_MASKING.md](/C:/TFG/TFGpractice/TRY14_TRY22_REEVALUATION_AND_MASKING.md)
  - explains why older `NLoS` results looked better
  - explains why excluding buildings changes the apparent results so much
  - important when comparing old and new tries

## Best documents for a supervisor

If the reader is a supervisor and wants the concise but complete story, these are the best choices:

- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)
  - best single-file overview

- [FOR_GENIA_SUMMARY_TRIES_20_TO_47.md](/C:/TFG/TFGpractice/FOR_GENIA_SUMMARY_TRIES_20_TO_47.md)
  - shorter supervisor-facing version focused on the recent family

- [PAPER_SOURCES_TRIES_20_TO_47.md](/C:/TFG/TFGpractice/PAPER_SOURCES_TRIES_20_TO_47.md)
  - source-to-try mapping
  - useful when the question is "why did we try this?"

- [TRY49_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY49_PRIOR_IMPROVEMENTS.md)
  - good short operational note for the current cluster-facing branch

## Best documents for implementation work

If the reader wants to change code or reproduce the latest setup, these are the most useful files:

- [TRY47_UNET22_PRIOR_NLOS_MOE.md](/C:/TFG/TFGpractice/TRY47_UNET22_PRIOR_NLOS_MOE.md)
  - architecture and training logic

- [TRY49_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY49_PRIOR_IMPROVEMENTS.md)
  - active stage1/stage2 path
  - best entry point for current PMNet implementation work

- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)
  - active prior-research note
  - points to archived failed prior experiments and the current baseline

- [TFGFiftyFirstTry51/README.md](/C:/TFG/TFGpractice/TFGFiftyFirstTry51/README.md)
  - active implementation note for `Try 51`
  - best entry point for the new literature-aligned branch

- [TFGFiftyFirstTry51/PRIOR_AGGREGATION_NOTE.md](/C:/TFG/TFGpractice/TFGFiftyFirstTry51/PRIOR_AGGREGATION_NOTE.md)
  - explains why `Try 51` prior overall RMSE can look better even when the
    regime-specific prior numbers do not both improve

- [TFGFiftyFourthTry54/README.md](/C:/TFG/TFGpractice/TFGFiftyFourthTry54/README.md)
  - active implementation note for the partitioned-expert branch

- [TFGFiftyFourthTry54/TRY54_IMPLEMENTATION_NOTES.md](/C:/TFG/TFGpractice/TFGFiftyFourthTry54/TRY54_IMPLEMENTATION_NOTES.md)
  - practical note for `PMHHNet`, `no_data`, validation JSON contents, and cluster intent

- [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)
  - calibration workflow and why it must be regenerated for a new dataset

- [TRY45_ENHANCED_PRIOR_AND_MOE.md](/C:/TFG/TFGpractice/TRY45_ENHANCED_PRIOR_AND_MOE.md)
  - the immediate precursor in the prior/MoE line

- [TRY46_LOS_NLOS_BRANCHING_AND_NLOS_EXPERTS.md](/C:/TFG/TFGpractice/TRY46_LOS_NLOS_BRANCHING_AND_NLOS_EXPERTS.md)
  - the branch that motivated explicit `LoS / NLoS` specialization

## Best documents for the new diagrams

If you want the visual map of the Try 50 to Try 54 branch family, start here:

- [Try 50 diagram page](/C:/TFG/TFGpractice/diagram/try50/index.html)
  - browser-friendly Mermaid page for the Try 50 prior system

- [Try 51 diagram page](/C:/TFG/TFGpractice/diagram/try51/index.html)
  - browser-friendly Mermaid page for the supervised PMNet baseline

- [Try 52 diagram page](/C:/TFG/TFGpractice/diagram/try52/index.html)
  - browser-friendly Mermaid page for the morphology-routed MoE branch

- [Try 53 diagram page](/C:/TFG/TFGpractice/diagram/try53/index.html)
  - browser-friendly Mermaid page for the cyclic feedback chain

- [Try 54 diagram page](/C:/TFG/TFGpractice/diagram/try54/index.html)
  - browser-friendly Mermaid page for the partitioned experts and PMHHNet

- [TRY51_TO_TRY54_DIAGRAMS.md](/C:/TFG/TFGpractice/TRY51_TO_TRY54_DIAGRAMS.md)
  - source text companion with direct links to the Mermaid files

## Best documents for understanding why PMNet was not enough

These are the most useful if the question is:

- why did the project explore PMNet?
- why did it not simply stay with PMNet?

Relevant documents:

- [TRY42_PMNET_RESIDUAL_ARCHITECTURE.md](/C:/TFG/TFGpractice/TRY42_PMNET_RESIDUAL_ARCHITECTURE.md)
- [TRY42_SOURCES_AND_PMNET_SCHEMA.md](/C:/TFG/TFGpractice/TRY42_SOURCES_AND_PMNET_SCHEMA.md)
- [TRY43_TRY44_PMNET_CONTROLS.md](/C:/TFG/TFGpractice/TRY43_TRY44_PMNET_CONTROLS.md)
- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)

The last one is especially useful because it explains the final lesson:

- PMNet with prior helped,
- PMNet without prior did not convince enough,
- and the project now wants to test whether the stronger U-Net spatial family was the better long-term base.

## Best documents for the spread branch

The spread branch has fewer current active experiments, but these files matter:

- [TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md](/C:/TFG/TFGpractice/TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md)
  - visual diagnosis of why spread prediction was underperforming

- [VERSIONS.md](/C:/TFG/TFGpractice/VERSIONS.md)
  - to identify which tries correspond to spread-specific branches

Historical spread highlights include:

- `Try 23`
- `Try 26`
- `Try 30`
- `Try 32`
- `Try 35`
- `Try 36`
- `Try 39`
- `Try 40`

## Best documents for the masking change

If the question is:

- what changed when buildings stopped counting?
- why did some old results stop looking so good?

the right documents are:

- [TRY14_TRY22_REEVALUATION_AND_MASKING.md](/C:/TFG/TFGpractice/TRY14_TRY22_REEVALUATION_AND_MASKING.md)
- [TRIES_33_TO_36_PHYSICAL_PRIORS_AND_BUILDING_MASK.md](/C:/TFG/TFGpractice/TRIES_33_TO_36_PHYSICAL_PRIORS_AND_BUILDING_MASK.md)
- [TRIES_37_TO_40_NEW_DATASET_AND_MASK_VERIFICATION.md](/C:/TFG/TFGpractice/TRIES_37_TO_40_NEW_DATASET_AND_MASK_VERIFICATION.md)

These together explain:

- the metric shift,
- the code-side masking change,
- and why modern `NLoS` is harder than historical `NLoS`.

## Best documents for physical priors and formulas

If the main question is:

- where does the physical prior come from?
- how is it calibrated?
- why is the calibration JSON saved?

the best reading set is:

- [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)
- [TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md](/C:/TFG/TFGpractice/TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md)
- [TRY45_RELEASE_GATE_AND_PRIOR_STATUS.md](/C:/TFG/TFGpractice/TRY45_RELEASE_GATE_AND_PRIOR_STATUS.md)
- [TRY47_UNET22_PRIOR_NLOS_MOE.md](/C:/TFG/TFGpractice/TRY47_UNET22_PRIOR_NLOS_MOE.md)

These explain:

- the raw prior,
- the calibrated prior,
- the release gate on `NLoS`,
- and why the calibrated JSON is part of the try definition.

For the current prior-refinement discussion, also read:

- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)
- [TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md](/C:/TFG/TFGpractice/TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md)

## Best documents for source tracing

If the goal is not experimental status but literature justification, use:

- [PAPER_SOURCES_TRIES_20_TO_47.md](/C:/TFG/TFGpractice/PAPER_SOURCES_TRIES_20_TO_47.md)
- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)

That pairing is useful because:

- the paper-sources file is the clean source map,
- while the master summary explains how those sources changed actual decisions.

## Branch-specific notes that are still useful

These are older focused notes that still matter because they preserve the reasoning behind specific family transitions:

- [TRY29_VISUAL_REVIEW_AND_RADIAL_PLAN.md](/C:/TFG/TFGpractice/TRY29_VISUAL_REVIEW_AND_RADIAL_PLAN.md)
- [TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md](/C:/TFG/TFGpractice/TRY30_SPREAD_VISUAL_REVIEW_AND_PLAN.md)
- [TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md](/C:/TFG/TFGpractice/TRY41_PRIOR_RESIDUAL_AND_REGIME_ANALYSIS.md)
- [TRY42_PMNET_RESIDUAL_ARCHITECTURE.md](/C:/TFG/TFGpractice/TRY42_PMNET_RESIDUAL_ARCHITECTURE.md)
- [TRY42_SOURCES_AND_PMNET_SCHEMA.md](/C:/TFG/TFGpractice/TRY42_SOURCES_AND_PMNET_SCHEMA.md)
- [TRY43_TRY44_PMNET_CONTROLS.md](/C:/TFG/TFGpractice/TRY43_TRY44_PMNET_CONTROLS.md)
- [TRY45_ENHANCED_PRIOR_AND_MOE.md](/C:/TFG/TFGpractice/TRY45_ENHANCED_PRIOR_AND_MOE.md)
- [TRY45_RELEASE_GATE_AND_PRIOR_STATUS.md](/C:/TFG/TFGpractice/TRY45_RELEASE_GATE_AND_PRIOR_STATUS.md)
- [TRY46_LOS_NLOS_BRANCHING_AND_NLOS_EXPERTS.md](/C:/TFG/TFGpractice/TRY46_LOS_NLOS_BRANCHING_AND_NLOS_EXPERTS.md)
- [TRY49_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY49_PRIOR_IMPROVEMENTS.md)
- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)

## Historical documents that remain useful mainly for traceability

These are not the best starting points for the current state, but they still preserve useful context:

- [CITY_REGIME_TRIES_15_TO_20.md](/C:/TFG/TFGpractice/CITY_REGIME_TRIES_15_TO_20.md)
- [IMPLEMENTED_NEW_TRIES_20_TO_22.md](/C:/TFG/TFGpractice/IMPLEMENTED_NEW_TRIES_20_TO_22.md)
- [TRIES_24_TO_28_IMPLEMENTATION.md](/C:/TFG/TFGpractice/TRIES_24_TO_28_IMPLEMENTATION.md)
- [PARADIGM_SHIFT_PLAN_TRIES_31_TO_33.md](/C:/TFG/TFGpractice/PARADIGM_SHIFT_PLAN_TRIES_31_TO_33.md)
- [TRIES_COMMON_CHANGES_AND_NEXT_TRIES.md](/C:/TFG/TFGpractice/TRIES_COMMON_CHANGES_AND_NEXT_TRIES.md)

These matter mostly when someone wants to reconstruct the exact sequence of local decisions over time.

## If the question is "what should I read for..."

### "...the current best narrative of the project?"

- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)

### "...the exact experiment chronology?"

- [VERSIONS.md](/C:/TFG/TFGpractice/VERSIONS.md)

### "...why old NLoS results looked better?"

- [TRY14_TRY22_REEVALUATION_AND_MASKING.md](/C:/TFG/TFGpractice/TRY14_TRY22_REEVALUATION_AND_MASKING.md)

### "...how the current prior system works?"

- [FORMULA_PRIOR_CALIBRATION_SYSTEM.md](/C:/TFG/TFGpractice/FORMULA_PRIOR_CALIBRATION_SYSTEM.md)

### "...what Try 47 actually is?"

- [TRY47_UNET22_PRIOR_NLOS_MOE.md](/C:/TFG/TFGpractice/TRY47_UNET22_PRIOR_NLOS_MOE.md)

### "...why the current two-ray prior is still too weak?"

- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)

### "...how should the LoS/NLoS prior be refined next?"

- [TRY50_PRIOR_IMPROVEMENTS.md](/C:/TFG/TFGpractice/TRY50_PRIOR_IMPROVEMENTS.md)
- [TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md](/C:/TFG/TFGpractice/TFGFiftiethTry50/FAILED_PRIOR_EXPERIMENTS_SUMMARY.md)

### "...which papers motivated the recent branches?"

- [PAPER_SOURCES_TRIES_20_TO_47.md](/C:/TFG/TFGpractice/PAPER_SOURCES_TRIES_20_TO_47.md)

### "...what to send a supervisor first?"

- [GENIA_AND_SOURCES_MASTER_SUMMARY.md](/C:/TFG/TFGpractice/GENIA_AND_SOURCES_MASTER_SUMMARY.md)
- [FOR_GENIA_SUMMARY_TRIES_20_TO_47.md](/C:/TFG/TFGpractice/FOR_GENIA_SUMMARY_TRIES_20_TO_47.md)

## Final note

The repository intentionally keeps historical notes instead of deleting them, because many experimental decisions only make sense when the intermediate failures are still visible.

However, for current work the safest rule is:

- start from the master summary,
- use `VERSIONS.md` for chronology,
- use the masking note to interpret old numbers,
- and use the active-branch note for the current implementation details.
