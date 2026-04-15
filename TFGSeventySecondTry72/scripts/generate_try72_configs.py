"""Generate Try 72 configs: 6 topology experts (single-stage 513x513) + classifier.

Key decisions documented in DATASET_ANALYSIS_AND_TRAINING_STRATEGY.md:

Resolution: 513x513 direct, NO Stage 2 refiner.
  - The tail refiner only improved 0.2-0.5 dB in past tries (Try 49-64).
  - With uint8 GT (1 dB resolution), the refiner is learning corrections
    smaller than the quantization noise.
  - Single-stage at full resolution is simpler and avoids resolution loss.

Experts: 6 topology classes (proven since Try 54), NOT split by LoS/NLoS or height.
  - LoS/NLoS is a per-pixel property, not per-sample. Try 46 tried this and it failed.
  - Height is handled continuously by FiLM with sinusoidal encoding; splitting would
    reduce data per expert and increase overfitting risk.
  - Topology class is the physically meaningful partition: building density and height
    determine the ray-tracing complexity.

Anti-overfitting: CutMix, higher dropout, higher weight decay, gradient accumulation.
  - CutMix (Yun et al., ICCV 2019) is the strongest single regularizer for limited
    dense-prediction datasets (confirmed by Zhang et al., MICCAI 2024).
  - batch_size=1 with gradient_accumulation_steps=16 (matches Slurm multi-GPU presets).

NLoS convergence: pixel-level NLoS reweighting in the loss.
  - LoS:NLoS valid pixel ratio is ~10:1. NLoS pixels get 4x weight (nlos_reweight_factor).

DataLoader defaults match dataloader_batch_presets/presets/4gpu.yml (also used for 2gpu/3gpu_medium copies).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Topology experts (same 6 classes proven since Try 54)
# ---------------------------------------------------------------------------
EXPERT_SPECS = [
    {"expert_id": "open_sparse_lowrise",   "topology_class": "open_sparse_lowrise"},
    {"expert_id": "open_sparse_vertical",  "topology_class": "open_sparse_vertical"},
    {"expert_id": "mixed_compact_lowrise", "topology_class": "mixed_compact_lowrise"},
    {"expert_id": "mixed_compact_midrise", "topology_class": "mixed_compact_midrise"},
    {"expert_id": "dense_block_midrise",   "topology_class": "dense_block_midrise"},
    {"expert_id": "dense_block_highrise",  "topology_class": "dense_block_highrise"},
]

MODEL_SPECS = {
    "open_sparse_lowrise":   {"base_channels": 20, "hf_channels": 10},
    "open_sparse_vertical":  {"base_channels": 20, "hf_channels": 10},
    "mixed_compact_lowrise": {"base_channels": 20, "hf_channels": 10},
    "mixed_compact_midrise": {"base_channels": 20, "hf_channels": 10},
    "dense_block_midrise":   {"base_channels": 20, "hf_channels": 10},
    "dense_block_highrise":  {"base_channels": 20, "hf_channels": 10},
}

TOPO_PART = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}


def _base_config(root: Path) -> dict:
    path = (
        root.parent
        / "TFGFiftyFirstTry51"
        / "experiments"
        / "fiftyfirsttry51_pmnet_prior_gan_fastbatch"
        / "fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml"
    )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cleanup(out_dir: Path, pattern: str) -> None:
    for stale in out_dir.glob(pattern):
        try:
            stale.unlink()
        except PermissionError:
            pass


def _apply_common_data(cfg: dict, topology_class: str) -> None:
    cfg["data"]["partition_filter"] = {"topology_class": topology_class}
    cfg["data"].pop("los_sample_filter", None)
    cfg["data"]["topology_partitioning"] = dict(TOPO_PART)
    # Aligned with experiments/dataloader_batch_presets/presets/4gpu.yml (and 2gpu / 3gpu_medium copies).
    cfg["data"]["num_workers"] = 4
    cfg["data"]["val_num_workers"] = 2
    cfg["data"]["persistent_workers"] = True
    cfg["data"]["val_persistent_workers"] = True
    cfg["data"]["prefetch_factor"] = 4
    cfg["data"]["exclude_non_ground_targets"] = True
    cfg["data"]["path_loss_saturation_db"] = 180
    cfg["data"]["path_loss_ignore_nonfinite"] = True
    cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
    cfg["data"]["derive_no_data_from_non_ground"] = True

    formula = cfg["data"].setdefault("path_loss_formula_input", {})
    formula["enabled"] = True
    formula["include_confidence_channel"] = True
    formula["cache_enabled"] = False
    formula["cache_dir"] = "prior_cache/try72_prior_auto_city_v1"
    formula["cache_version"] = "try72_prior_auto_city_v1"

    cfg["data"]["tx_depth_map_channel"] = True
    cfg["data"]["elevation_angle_map_channel"] = True
    cfg["data"]["building_mask_channel"] = True

    cfg["data"]["receiver_subsample"] = {
        "enabled": True,
        "keep_fraction": 0.01,
        "tile_side_m": 1000.0,
        "max_rx_per_tile": 32,
        "val_seed": 42,
    }

    obs = cfg["data"].setdefault("path_loss_obstruction_features", {})
    obs["enabled"] = False
    obs["include_shadow_depth"] = False
    obs["include_distance_since_los_break"] = False
    obs["include_max_blocker_height"] = False
    obs["include_blocker_count"] = False

    cfg["augmentation"] = {
        "enable": True,
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rot90_prob": 0.5,
    }


def _build_expert(cfg: dict, spec: dict, model_spec: dict) -> dict:
    topology_class = spec["topology_class"]
    _apply_common_data(cfg, topology_class)

    # --- Resolution: 513x513 direct, single stage ---
    cfg["data"]["image_size"] = 513
    cfg["data"]["val_batch_size"] = 4

    # --- Training recipe (matches try72_expert_*.yaml in repo) ---
    t = cfg["training"]
    t["batch_size"] = 4
    t["gradient_accumulation_steps"] = 4
    t["epochs"] = 1000
    t["optimizer"] = "adamw"
    t["discriminator_optimizer"] = "adam"
    t["learning_rate"] = 8.0e-4
    t["discriminator_lr"] = 9.0e-5
    t["weight_decay"] = 0.1
    t["beta1"] = 0.5
    t["beta2"] = 0.999

    t["ema_decay"] = 0.975

    t["lr_scheduler"] = "cosine_annealing_lr"
    t["lr_scheduler_factor"] = 0.5
    t["lr_scheduler_patience"] = 5
    t["lr_warmup_optimizer_steps"] = 0
    t["lr_warmup_start_factor"] = 0.5
    t["lr_scheduler_T_max"] = 1000
    t["lr_scheduler_min_lr"] = 5.0e-6
    t["lr_scheduler_eta_min"] = 5.0e-4
    t["save_validation_json_each_epoch"] = True
    t["auto_batch_by_vram"] = {
        "enabled": False,
        "reference_vram_gb": 80.0,
        "reference_batch_size": 1,
        "min_batch_size": 1,
        "max_batch_size": 5,
        "safety_factor": 0.9,
    }
    t["amp"] = True
    t["clip_grad_norm"] = 1.0

    t["generator_objective"] = "full_map_rmse_only"
    t["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
    t["save_every"] = 5
    t["run_final_test_after_training"] = True

    t["early_stopping"] = {
        "enabled": True,
        "patience": 50,
        "min_delta": 0.0,
        "rewind_to_best_model": True,
    }

    # --- NLoS pixel reweighting ---
    t["nlos_reweight_factor"] = 4.0

    # --- CutMix ---
    t["cutmix_prob"] = 0.25
    t["cutmix_alpha"] = 1.0

    t["regime_reweighting"] = {"enabled": False}

    # --- Model ---
    m = cfg["model"]
    m["arch"] = "pmhhnet"
    m["base_channels"] = model_spec["base_channels"]
    m["hf_channels"] = model_spec["hf_channels"]
    m["disc_base_channels"] = 0
    m["out_channels"] = 1
    m["dropout"] = 0.12
    m["gradient_checkpointing"] = True
    m["use_scalar_film"] = True
    m["use_scalar_channels"] = True
    m["use_se_attention"] = True
    m["se_reduction"] = 4

    # --- Loss ---
    lo = cfg["loss"]
    lo["lambda_recon"] = 0.0
    lo["lambda_gan"] = 0.0
    lo["loss_type"] = "huber"
    lo["huber_delta"] = 0.14
    lo["huber_delta_normalized"] = True
    lo["full_map_generator_scale"] = 25.0
    lo["mse_weight"] = 0.0
    lo["l1_weight"] = 0.0

    cfg["multiscale_path_loss"] = {
        "enabled": True,
        "scales": [2, 4],
        "weights": [0.6, 0.4],
        "min_valid_ratio": 0.5,
        "loss_weight": 0.55,
    }

    cfg["prior_residual_path_loss"] = {
        "enabled": True,
        "use_formula_input_channel": True,
        "optimize_residual_only": False,
        "clamp_final_output": True,
        "loss_weight": 0.5,
        "mse_weight": 1.0,
        "l1_weight": 0.0,
        "final_loss_weight_when_residual_only": 0.0,
        "multiscale_loss_weight_when_residual_only": 0.0,
    }

    cfg["no_data_auxiliary"] = {
        "enabled": False,
        "loss_weight": 0.0,
        "positive_weight": 1.0,
    }

    cfg["nlos_focus_loss"] = {"enabled": False}

    cfg["corridor_weighting"] = {
        "enabled": True,
        "sigma": 40.0,
        "kappa": 150.0,
        "min_weight": 0.3,
    }

    cfg["test_time_augmentation"] = {
        "enabled": True,
        "transforms": "d4",
    }

    return cfg


def main() -> None:
    root = ROOT
    base_cfg = _base_config(root)

    expert_dir = root / "experiments" / "seventysecond_try72_experts"
    cl_dir = root / "experiments" / "seventysecond_try72_classifier"
    expert_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)

    _cleanup(expert_dir, "try72_expert_*.yaml")

    registry = {"experts": []}

    for spec in EXPERT_SPECS:
        tc = spec["topology_class"]
        ms = MODEL_SPECS[tc]

        cfg = _build_expert(deepcopy(base_cfg), spec, ms)
        name = f"try72_expert_{spec['expert_id']}"
        cfg["runtime"]["output_dir"] = f"outputs/{name}"
        cfg["runtime"]["resume_checkpoint"] = ""

        path = expert_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        registry["experts"].append({
            "expert_id": spec["expert_id"],
            "topology_class": tc,
            "config": str(path.relative_to(root)).replace("\\", "/"),
            "checkpoint": f"outputs/{name}/best_model.pt",
            "model_arch": "pmhhnet",
        })

    # --- Topology classifier ---
    cl_cfg = {
        "seed": 42,
        "data": {
            "hdf5_path": "../Datasets/CKM_Dataset_270326.h5",
            "input_column": "topology_map",
            "input_metadata": {"scale": 255.0, "offset": 0.0},
            "hdf5_scalar_specs": [
                {"name": "antenna_height_m", "from_dataset": "uav_height"},
            ],
            "non_ground_threshold": 0.0,
            "split_mode": "city_holdout",
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "split_seed": 42,
            "image_size": 257,
            "num_workers": 4,
            "path_loss_formula_input": {
                "regime_calibration_json": "prior_calibration/regime_obstruction_train_only_from_try47.json",
            },
            "topology_partitioning": dict(TOPO_PART),
            "scalar_feature_norms": {"antenna_height_m": 120.0},
        },
        "model": {
            "base_channels": 24,
            "dropout": 0.08,
            "norm_type": "group",
            "use_antenna_scalar": False,
        },
        "training": {
            "batch_size": 8,
            "epochs": 500,
            "learning_rate": 6.0e-4,
            "weight_decay": 1.0e-4,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 3,
            "lr_scheduler_min_lr": 1.0e-5,
            "clip_grad_norm": 1.0,
        },
        "runtime": {
            "device": "cuda",
            "output_dir": "outputs/try72_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    cl_path = cl_dir / "try72_topology_classifier.yaml"
    with cl_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cl_cfg, f, sort_keys=False)

    with (expert_dir / "try72_expert_registry.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    print(f"Generated {len(EXPERT_SPECS)} expert configs in {expert_dir}")
    print(f"Generated classifier config in {cl_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
