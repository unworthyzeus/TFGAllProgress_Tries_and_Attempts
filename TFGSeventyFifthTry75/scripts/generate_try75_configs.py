"""Generate Try 75 configs: two all-city experts, LoS-only and NLoS-only.

Try 75 is the height-aware continuation stage after the split-mask Try 74:

1. Reopen the full antenna-height range.
2. Stay prior-free.
3. Keep the hard supervision split:
   - one model trains only on LoS pixels
   - one model trains only on NLoS pixels
4. The LoS mask is loaded only to build the valid-loss mask, not as a model input.
5. Re-enable scalar height FiLM so height modulation becomes the main new
   mechanism relative to Try 74.
6. Default each model to resume from the matching Try 74 checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERT_SPECS = [
    {
        "expert_id": "allcity_los",
        "resume_from_try74_id": "band4555_allcity_los",
        "los_region_mask_mode": "los_only",
        "base_channels": 48,
        "hf_channels": 20,
    },
    {
        "expert_id": "allcity_nlos",
        "resume_from_try74_id": "band4555_allcity_nlos",
        "los_region_mask_mode": "nlos_only",
        "base_channels": 48,
        "hf_channels": 20,
    },
]


def _load_try67_template(root: Path, expert_id: str) -> dict:
    path = (
        root.parent
        / "TFGSixtySeventhTry67"
        / "experiments"
        / "sixtyseventh_try67_experts"
        / f"try67_expert_{expert_id}.yaml"
    )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cleanup(out_dir: Path, pattern: str) -> None:
    for stale in out_dir.glob(pattern):
        try:
            stale.unlink()
        except PermissionError:
            pass


def _build_expert(root: Path, spec: dict) -> dict:
    cfg = _load_try67_template(root, "mixed_midrise")

    cfg["data"]["partition_filter"] = {}
    cfg["data"].pop("los_sample_filter", None)
    cfg["data"].pop("topology_partitioning", None)
    cfg["data"]["los_input_column"] = "los_mask"
    cfg["data"]["use_los_as_input"] = False
    cfg["data"]["los_region_mask_mode"] = str(spec["los_region_mask_mode"])
    cfg["data"]["los_region_mask_threshold"] = 0.5

    formula = cfg["data"].setdefault("path_loss_formula_input", {})
    formula["enabled"] = False
    formula["include_confidence_channel"] = False
    formula["cache_enabled"] = False
    formula["cache_dir"] = "prior_cache/try75_no_prior_allheights"
    formula["cache_version"] = "try75_no_prior_allheights"
    formula["regime_calibration_json"] = "prior_calibration/regime_obstruction_train_only_from_try47.json"

    cfg["data"]["scalar_feature_columns"] = ["antenna_height_m"]
    cfg["data"]["tx_depth_map_channel"] = True
    cfg["data"]["elevation_angle_map_channel"] = True
    cfg["data"]["building_mask_channel"] = True
    cfg["data"]["val_batch_size"] = 1
    cfg["data"]["num_workers"] = 6
    cfg["data"]["val_num_workers"] = 4
    cfg["data"]["persistent_workers"] = True
    cfg["data"]["val_persistent_workers"] = True
    cfg["data"]["prefetch_factor"] = 4
    cfg["data"]["path_loss_ignore_nonfinite"] = True
    cfg["data"]["path_loss_saturation_db"] = 180
    cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
    cfg["data"]["derive_no_data_from_non_ground"] = True

    cfg["data"].pop("knife_edge_channel", None)
    obs = cfg["data"].setdefault("path_loss_obstruction_features", {})
    obs["enabled"] = True
    obs["include_shadow_depth"] = True
    obs["include_distance_since_los_break"] = True
    obs["include_max_blocker_height"] = True
    obs["include_blocker_count"] = True

    model_cfg = cfg.setdefault("model", {})
    model_cfg["arch"] = "pmhhnet"
    model_cfg["base_channels"] = int(spec["base_channels"])
    model_cfg["hf_channels"] = int(spec["hf_channels"])
    model_cfg["out_channels"] = 1
    model_cfg["use_scalar_channels"] = True
    model_cfg["use_scalar_film"] = True
    model_cfg["gradient_checkpointing"] = True
    model_cfg["dropout"] = 0.12
    model_cfg["use_se_attention"] = True
    model_cfg["se_reduction"] = 4
    model_cfg["absolute_output_bias_init_db"] = "auto"

    train_cfg = cfg.setdefault("training", {})
    train_cfg["batch_size"] = 1
    train_cfg["gradient_accumulation_steps"] = 8
    train_cfg["epochs"] = 800
    train_cfg["optimizer"] = "adamw"
    train_cfg["learning_rate"] = 8.0e-4
    train_cfg["weight_decay"] = 1.5e-2
    train_cfg["beta1"] = 0.9
    train_cfg["beta2"] = 0.999
    train_cfg["ema_decay"] = 0.95
    train_cfg["amp"] = True
    train_cfg["clip_grad_norm"] = 1.0
    train_cfg["save_every"] = 5
    train_cfg["generator_objective"] = "full_map_rmse_only"
    train_cfg["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
    train_cfg["run_final_test_after_training"] = True
    train_cfg["save_validation_json_each_epoch"] = True
    train_cfg["lr_scheduler"] = "reduce_on_plateau"
    train_cfg["lr_scheduler_factor"] = 0.5
    train_cfg["lr_scheduler_patience"] = 8
    train_cfg["lr_scheduler_min_lr"] = 1.0e-6
    train_cfg["lr_warmup_optimizer_steps"] = 250
    train_cfg["lr_warmup_start_factor"] = 0.2
    train_cfg["regime_reweighting"] = {"enabled": False}
    train_cfg["cutmix_prob"] = 0.25
    train_cfg["cutmix_alpha"] = 1.0
    train_cfg["nlos_reweight_factor"] = 1.0

    loss_cfg = cfg.setdefault("loss", {})
    loss_cfg["lambda_recon"] = 0.0
    loss_cfg["lambda_gan"] = 0.0
    loss_cfg["mse_weight"] = 0.0
    loss_cfg["l1_weight"] = 0.0
    loss_cfg["loss_type"] = "mse"
    loss_cfg["full_map_generator_scale"] = 25.0

    cfg["multiscale_path_loss"] = {
        "enabled": True,
        "scales": [2, 4],
        "weights": [0.6, 0.4],
        "min_valid_ratio": 0.5,
        "loss_weight": 0.55,
    }

    cfg["prior_residual_path_loss"] = {
        "enabled": False,
        "absolute_prediction": True,
        "use_formula_input_channel": False,
        "optimize_residual_only": False,
        "clamp_final_output": False,
        "loss_weight": 0.0,
        "mse_weight": 1.0,
        "l1_weight": 0.0,
        "final_loss_weight_when_residual_only": 0.0,
        "multiscale_loss_weight_when_residual_only": 0.0,
    }

    cfg["nlos_focus_loss"] = {"enabled": False}
    cfg["corridor_weighting"] = {"enabled": False, "sigma": 40.0, "kappa": 150.0, "min_weight": 0.3}
    cfg["no_data_auxiliary"] = {"enabled": False, "loss_weight": 0.0, "positive_weight": 1.0}

    tmeta = cfg.setdefault("target_metadata", {}).setdefault("path_loss", {})
    tmeta["scale"] = 180.0
    tmeta["offset"] = 0.0
    tmeta["unit"] = "dB"
    tmeta["clip_min"] = 0.0
    tmeta["clip_max"] = 180.0
    tmeta["predict_linear"] = False
    tmeta.pop("clip_min_db", None)
    tmeta.pop("clip_max_db", None)

    return cfg


def main() -> None:
    root = ROOT
    expert_dir = root / "experiments" / "seventyfifth_try75_experts"
    cl_dir = root / "experiments" / "seventyfifth_try75_classifier"
    expert_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)

    _cleanup(expert_dir, "try75_expert_*.yaml")
    _cleanup(expert_dir, "try75_expert_registry.yaml")

    registry = {"experts": []}
    for spec in EXPERT_SPECS:
        cfg = _build_expert(root, deepcopy(spec))
        name = f"try75_expert_{spec['expert_id']}"
        cfg["runtime"]["output_dir"] = f"outputs/{name}"
        cfg["runtime"]["resume_checkpoint"] = (
            f"../TFGSeventyFourthTry74/outputs/try74_expert_{spec['resume_from_try74_id']}/best_model.pt"
        )

        path = expert_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        registry["experts"].append({
            "expert_id": spec["expert_id"],
            "config": str(path.relative_to(root)).replace("\\", "/"),
            "checkpoint": f"../TFGSeventyFourthTry74/outputs/try74_expert_{spec['resume_from_try74_id']}/best_model.pt",
            "model_arch": "pmhhnet",
        })

    cl_cfg = {
        "seed": 42,
        "data": {
            "hdf5_path": "../Datasets/CKM_Dataset_270326.h5",
            "input_column": "topology_map",
            "input_metadata": {"scale": 255.0, "offset": 0.0},
            "hdf5_scalar_specs": [{"name": "antenna_height_m", "from_dataset": "uav_height"}],
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
        },
        "model": {"base_channels": 24, "dropout": 0.08, "norm_type": "group", "use_antenna_scalar": False},
        "training": {
            "batch_size": 8,
            "epochs": 400,
            "learning_rate": 6.0e-4,
            "weight_decay": 1.0e-4,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 3,
            "lr_scheduler_min_lr": 1.0e-5,
            "clip_grad_norm": 1.0,
        },
        "runtime": {"device": "cuda", "output_dir": "outputs/try75_topology_classifier", "resume_checkpoint": ""},
    }
    cl_path = cl_dir / "try75_topology_classifier.yaml"
    with cl_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cl_cfg, f, sort_keys=False)

    with (expert_dir / "try75_expert_registry.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    print(f"Generated {len(EXPERT_SPECS)} expert configs in {expert_dir}")
    print(f"Generated classifier config in {cl_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
