# Try 50 to Try 54 Diagrams

This file is the source companion to the per-try browser pages under
[diagram/try50/](C:/TFG/TFGpractice/diagram/try50/index.html),
[diagram/try51/](C:/TFG/TFGpractice/diagram/try51/index.html),
[diagram/try52/](C:/TFG/TFGpractice/diagram/try52/index.html),
[diagram/try53/](C:/TFG/TFGpractice/diagram/try53/index.html), and
[diagram/try54/](C:/TFG/TFGpractice/diagram/try54/index.html).
Use the HTML pages when you want the Mermaid-rendered view; keep this file when
you want the short text index and direct links to the .mmd sources.

It stays useful as a textual overview of the recent path-loss branches and as a
lightweight companion to [TFGFiftyFourthTry54/README.md](C:/TFG/TFGpractice/TFGFiftyFourthTry54/README.md)
and [PMNET_VS_PMHHNET.md](C:/TFG/TFGpractice/TFGFiftyFourthTry54/PMNET_VS_PMHHNET.md).

## Try 50: structured prior

- [try50_prior_system.mmd](C:/TFG/TFGpractice/diagram/try50/try50_prior_system.mmd)
	- coherent LoS branch with damped two-ray smoothing and shadowed ripple
	- structured NLoS branch with shadow-depth, blocker-severity, and urban-context losses
	- confidence-aware formula channel for the prior input stack

## Try 51: supervised PMNet baseline

- [try51_full_system.mmd](C:/TFG/TFGpractice/diagram/try51/try51_full_system.mmd)
	- supervised PMNet stage 1
	- regime-aware weighting that boosts `NLoS`, low antenna, and dense high-rise cases
	- residual tail refiner on top of the best stage 1 checkpoint

## Try 52: morphology-routed MoE

- [try52_full_system.mmd](C:/TFG/TFGpractice/diagram/try52/try52_full_system.mmd)
	- automatic city-type routing from topology statistics
	- city-routed `NLoS` MoE with a shared PMNet trunk
	- lighter stage 2 refiner and `NLoS`-only stage 3 global-context refinement

## Try 53: cyclic feedback chain

- [try53_cyclic_system.mmd](C:/TFG/TFGpractice/diagram/try53/try53_cyclic_system.mmd)
	- bootstrap from the Try 51 best stage 1 checkpoint
	- stage 2 and stage 3 train in a forward chain
	- validation JSON feeds back into stage 1 weighting and resume cycles

## Try 54: partitioned experts and PMHHNet

- [try54_partitioned_system.mmd](C:/TFG/TFGpractice/diagram/try54/try54_partitioned_system.mmd)
	- topology-class routing with one expert per partition
	- six topology classes and an expert registry
	- shared PMHHNet expert family with continuous height conditioning
	- auxiliary `no_data` prediction alongside path-loss residual regression

- [pmhhnet_model.mmd](C:/TFG/TFGpractice/diagram/try54/pmhhnet_model.mmd)
	- PMHHNet architecture detail
	- high-frequency branch, FiLM-style height conditioning, and dual output head
