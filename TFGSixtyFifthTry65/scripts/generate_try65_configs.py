from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRY65_EXPERT_SPECS = [
    {"expert_id": "open_sparse_lowrise", "topology_class": "open_sparse_lowrise"},
    {"expert_id": "open_sparse_vertical", "topology_class": "open_sparse_vertical"},
    {"expert_id": "mixed_compact_lowrise", "topology_class": "mixed_compact_lowrise"},
    {"expert_id": "mixed_compact_midrise", "topology_class": "mixed_compact_midrise"},
    {"expert_id": "dense_block_midrise", "topology_class": "dense_block_midrise"},
    {"expert_id": "dense_block_highrise", "topology_class": "dense_block_highrise"},
]

TOPOLOGY_MODEL_SPECS = {
    "open_sparse_lowrise": {"base_channels": 20},
    "open_sparse_vertical": {"base_channels": 20},
    "mixed_compact_lowrise": {"base_channels": 20},
    "mixed_compact_midrise": {"base_channels": 20},
    "dense_block_midrise": {"base_channels": 20},
    "dense_block_highrise": {"base_channels": 20},
}


def _base_stage1_config(root: Path) -> dict:
    path = (
        root.parent
        / "TFGFiftyFirstTry51"
        / "experiments"
        / "fiftyfirsttry51_pmnet_prior_gan_fastbatch"
        / "fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml"
    )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _cleanup_existing(out_dir: Path, pattern: str) -> None:
    for stale in out_dir.glob(pattern):
        try:
            stale.unlink()
        except PermissionError:
            pass


def main() -> None:
    root = ROOT
    stage1_base = _base_stage1_config(root)

    stage1_dir = root / "experiments" / "sixtyfifthtry65_grokking_stage1"
    classifier_dir = root / "experiments" / "sixtyfifthtry65_topology_classifier"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_existing(stage1_dir, "sixtyfifthtry65_expert_*.yaml")

    stage1_registry = {"experts": []}

    for spec in TRY65_EXPERT_SPECS:
        topology_class = str(spec["topology_class"])
        model_spec = TOPOLOGY_MODEL_SPECS[topology_class]

        cfg = deepcopy(stage1_base)
        cfg["data"]["partition_filter"] = {"topology_class": topology_class}
        cfg["data"].pop("los_sample_filter", None)
        cfg["data"]["topology_partitioning"] = {
            "density_q1": 0.12,
            "density_q2": 0.28,
            "height_q1": 12.0,
            "height_q2": 28.0,
        }
        cfg["data"]["image_size"] = 513
        cfg["data"]["num_workers"] = 6
        cfg["data"]["val_num_workers"] = 2
        cfg["data"]["val_batch_size"] = 1
        cfg["data"]["persistent_workers"] = True
        cfg["data"]["val_persistent_workers"] = False
        cfg["data"]["prefetch_factor"] = 4
        cfg["data"]["exclude_non_ground_targets"] = True
        cfg["data"]["path_loss_saturation_db"] = 180
        cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
        cfg["data"]["derive_no_data_from_non_ground"] = True

        formula_cfg = cfg["data"].setdefault("path_loss_formula_input", {})
        formula_cfg["enabled"] = False
        formula_cfg["include_confidence_channel"] = False
        formula_cfg["cache_enabled"] = False
        formula_cfg["cache_dir"] = "prior_cache/try65_prior_auto_city_v1"
        formula_cfg["cache_version"] = "try65_prior_auto_city_v1"

        obstruction_cfg = cfg["data"].setdefault("path_loss_obstruction_features", {})
        obstruction_cfg["enabled"] = False
        obstruction_cfg["precomputed_hdf5"] = "../TFGFiftiethTry50/precomputed/obstruction_features_u8.h5"
        obstruction_cfg["meters_per_pixel"] = 1.0
        obstruction_cfg["angle_bins"] = 720
        obstruction_cfg["include_shadow_depth"] = False
        obstruction_cfg["include_distance_since_los_break"] = False
        obstruction_cfg["include_max_blocker_height"] = False
        obstruction_cfg["include_blocker_count"] = False

        cfg["augmentation"] = {
            "enable": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rot90_prob": 0.5,
        }
        cfg["training"]["batch_size"] = 1
        cfg["training"]["epochs"] = 10000
        cfg["training"]["optimizer"] = "adam"
        cfg["training"]["learning_rate"] = 1.0e-3
        cfg["training"]["weight_decay"] = 5.0e-2
        cfg["training"]["ema_decay"] = 1.0
        cfg["training"]["lr_scheduler"] = "none"
        cfg["training"]["generator_objective"] = "full_map_rmse_only"
        cfg["training"]["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
        cfg["training"]["save_every"] = 25
        cfg["training"]["run_final_test_after_training"] = True
        cfg["training"]["early_stopping"] = {
            "enabled": False,
            "patience": 0,
            "min_delta": 0.0,
            "rewind_to_best_model": False,
        }
        cfg["training"]["regime_reweighting"] = {
            "enabled": False,
            "los_weight": 1.0,
            "nlos_weight": 1.0,
        }

        cfg["model"]["base_channels"] = model_spec["base_channels"]
        cfg["model"]["disc_base_channels"] = 0
        cfg["model"]["out_channels"] = 1
        cfg["model"]["dropout"] = 0.05
        cfg["model"]["gradient_checkpointing"] = False

        cfg["loss"]["lambda_recon"] = 0.0
        cfg["loss"]["lambda_gan"] = 0.0
        cfg["loss"]["mse_weight"] = 1.0
        cfg["loss"]["l1_weight"] = 0.0
        cfg["multiscale_path_loss"] = {
            "enabled": False,
            "scales": [2, 4],
            "weights": [0.6, 0.4],
            "min_valid_ratio": 0.5,
            "loss_weight": 0.0,
        }
        cfg["prior_residual_path_loss"] = {
            "enabled": False,
            "use_formula_input_channel": False,
            "optimize_residual_only": False,
            "clamp_final_output": True,
            "loss_weight": 0.0,
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
        cfg["nlos_focus_loss"] = {
            "enabled": False,
            "mode": "rmse",
            "loss_weight": 0.0,
            "selection_alpha": 0.0,
        }

        stage1_name = f"sixtyfifthtry65_expert_{spec['expert_id']}"
        cfg["runtime"]["output_dir"] = f"outputs/{stage1_name}"
        cfg["runtime"]["resume_checkpoint"] = ""

        cfg_path = stage1_dir / f"{stage1_name}.yaml"
        with cfg_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        stage1_registry["experts"].append(
            {
                "expert_id": str(spec["expert_id"]),
                "topology_class": topology_class,
                "config": str(cfg_path.relative_to(root)).replace("\\", "/"),
                "checkpoint": f"outputs/{stage1_name}/latest_model.pt",
                "model_arch": "pmhhnet",
            }
        )

    classifier_cfg = {
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
            "topology_partitioning": {
                "density_q1": 0.12,
                "density_q2": 0.28,
                "height_q1": 12.0,
                "height_q2": 28.0,
            },
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
            "learning_rate": 8.0e-4,
            "weight_decay": 1.0e-4,
            "lr_scheduler_factor": 0.5,
            "lr_scheduler_patience": 3,
            "lr_scheduler_min_lr": 1.0e-5,
            "clip_grad_norm": 1.0,
        },
        "runtime": {
            "device": "cuda",
            "output_dir": "outputs/sixtyfifthtry65_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    classifier_path = classifier_dir / "sixtyfifthtry65_topology_classifier.yaml"
    with classifier_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(classifier_cfg, handle, sort_keys=False)

    with (stage1_dir / "sixtyfifthtry65_expert_registry.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(stage1_registry, handle, sort_keys=False)


if __name__ == "__main__":
    main()
