# Try 69 — Try 67 + SOA engineering gaps closed

`Try 69` is a **fork of Try 67** (`TFGpractice/scripts/bootstrap_try69_from_try67.py`) with the same **3 morphology experts**, dataset, knife-edge channel, PDE loss, and PMHHNet stem+HF path. It adds items that the Try 67 **SOA checklist** still marked as missing or off by default:

1. **Dual LoS/NLoS residual heads** — `model.out_channels: 2`, `dual_los_nlos_head.enabled: true`, blended in the trainer by `los_mask`.
2. **Corridor loss weights** — radial map centred on the UAV (map centre), multiplied into the **training** loss mask (`corridor_weighting` + `_corridor_spatial_multiplier` in `train_partitioned_pathloss_expert.py`). Standard validation RMSE remains unweighted for comparability.
3. **D4 TTA on validation** — `test_time_augmentation.use_in_validation: true` (still on for final test).

**Still not in scope** (would be separate research stacks): test-time adaptation / calibration, GAN default-on training, diffusion refiner, FM-RME pre-training, RadioPiT-style transformer.

Config generator: `scripts/generate_try69_configs.py`. Cluster layout mirrors Try 67 with renamed paths (`try69_expert_*`, `run_sixtyninth_try69_*.slurm`, `MASTER_PORT` default **29976**, submitter `--base-master-port` default **30286**).

See `SOA_IMPLEMENTATION_STATUS.md` in this folder for the Try 69–specific checklist.
