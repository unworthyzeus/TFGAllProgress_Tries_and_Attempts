from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRY63_EXPERT_SPECS = [
    {"expert_id": "open_sparse_lowrise", "topology_class": "open_sparse_lowrise"},
    {"expert_id": "open_sparse_vertical", "topology_class": "open_sparse_vertical"},
    {"expert_id": "mixed_compact_lowrise", "topology_class": "mixed_compact_lowrise"},
    {"expert_id": "mixed_compact_midrise", "topology_class": "mixed_compact_midrise"},
    {"expert_id": "dense_block_midrise", "topology_class": "dense_block_midrise"},
    {"expert_id": "dense_block_highrise", "topology_class": "dense_block_highrise"},
]

TOPOLOGY_MODEL_SPECS = {
    "open_sparse_lowrise": {"base_channels": 32, "refiner_channels": 32},
    "open_sparse_vertical": {"base_channels": 32, "refiner_channels": 32},
    "mixed_compact_lowrise": {"base_channels": 32, "refiner_channels": 32},
    "mixed_compact_midrise": {"base_channels": 32, "refiner_channels": 32},
    "dense_block_midrise": {"base_channels": 32, "refiner_channels": 32},
    "dense_block_highrise": {"base_channels": 32, "refiner_channels": 32},
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


def _base_stage2_config(root: Path) -> dict:
    path = (
        root.parent
        / "TFGFiftyFirstTry51"
        / "experiments"
        / "fiftyfirsttry51_pmnet_tail_refiner_fastbatch"
        / "fiftyfirsttry51_pmnet_tail_refiner_stage2.yaml"
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
    stage2_base = _base_stage2_config(root)

    stage1_dir = root / "experiments" / "sixtythirdtry63_partitioned_stage1"
    stage2_dir = root / "experiments" / "sixtythirdtry63_tail_refiner_stage2"
    classifier_dir = root / "experiments" / "sixtythirdtry63_topology_classifier"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_existing(stage1_dir, "sixtythirdtry63_expert_*.yaml")
    _cleanup_existing(stage2_dir, "sixtythirdtry63_tail_refiner_*.yaml")

    stage1_registry = {"experts": []}
    stage2_registry = {"experts": []}

    for spec in TRY63_EXPERT_SPECS:
        topology_class = str(spec["topology_class"])
        model_spec = TOPOLOGY_MODEL_SPECS[topology_class]

        stage1_cfg = deepcopy(stage1_base)
        stage1_cfg["data"]["partition_filter"] = {"topology_class": topology_class}
        stage1_cfg["data"].pop("los_sample_filter", None)
        stage1_cfg["data"]["topology_partitioning"] = {
            "density_q1": 0.12,
            "density_q2": 0.28,
            "height_q1": 12.0,
            "height_q2": 28.0,
        }
        stage1_cfg["data"]["image_size"] = 128
        stage1_cfg["data"]["num_workers"] = 6
        stage1_cfg["data"]["val_num_workers"] = 2
        stage1_cfg["data"]["val_batch_size"] = 6
        stage1_cfg["data"]["persistent_workers"] = True
        stage1_cfg["data"]["val_persistent_workers"] = False
        stage1_cfg["data"]["prefetch_factor"] = 4
        stage1_cfg["data"]["exclude_non_ground_targets"] = True
        stage1_cfg["data"]["path_loss_saturation_db"] = 180
        stage1_cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
        stage1_cfg["data"]["derive_no_data_from_non_ground"] = True
        formula_cfg = stage1_cfg["data"].setdefault("path_loss_formula_input", {})
        formula_cfg["enabled"] = True
        formula_cfg["include_confidence_channel"] = True
        formula_cfg["cache_enabled"] = False
        formula_cfg["cache_dir"] = "prior_cache/try63_prior_auto_city_v1"
        formula_cfg["cache_version"] = "try63_prior_auto_city_v1"
        obstruction_cfg = stage1_cfg["data"].setdefault("path_loss_obstruction_features", {})
        obstruction_cfg["enabled"] = True
        obstruction_cfg["precomputed_hdf5"] = "../TFGFiftiethTry50/precomputed/obstruction_features_u8.h5"
        obstruction_cfg["meters_per_pixel"] = 1.0
        obstruction_cfg["angle_bins"] = 720
        obstruction_cfg["include_shadow_depth"] = True
        obstruction_cfg["include_distance_since_los_break"] = True
        obstruction_cfg["include_max_blocker_height"] = True
        obstruction_cfg["include_blocker_count"] = True
        stage1_cfg["augmentation"] = {
            "enable": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rot90_prob": 0.5,
        }
        stage1_cfg["training"]["batch_size"] = 6
        stage1_cfg["training"]["epochs"] = 250
        stage1_cfg["training"]["optimizer"] = "adam"
        stage1_cfg["training"]["learning_rate"] = 8e-5
        stage1_cfg["training"]["weight_decay"] = 0.0
        stage1_cfg["training"]["ema_decay"] = 1.0
        stage1_cfg["training"]["lr_scheduler"] = "reduce_on_plateau"
        stage1_cfg["training"]["lr_scheduler_factor"] = 0.5
        stage1_cfg["training"]["lr_scheduler_patience"] = 5
        stage1_cfg["training"]["lr_scheduler_min_lr"] = 5e-6
        stage1_cfg["training"]["generator_objective"] = "full_map_rmse_only"
        stage1_cfg["training"]["selection_metrics"] = {"path_loss_513.rmse_physical": 1.0}
        stage1_cfg["training"]["save_every"] = 1
        stage1_cfg["training"]["run_final_test_after_training"] = True
        stage1_cfg["training"]["early_stopping"] = {
            "enabled": True,
            "patience": 6,
            "min_delta": 0.0,
            "rewind_to_best_model": True,
        }
        stage1_cfg["training"]["regime_reweighting"] = {
            "enabled": False,
            "los_weight": 1.0,
            "nlos_weight": 1.0,
        }
        stage1_cfg["model"]["base_channels"] = model_spec["base_channels"]
        stage1_cfg["model"]["disc_base_channels"] = 0
        stage1_cfg["model"]["out_channels"] = 1
        stage1_cfg["model"]["dropout"] = 0.05
        stage1_cfg["loss"]["lambda_recon"] = 0.0
        stage1_cfg["loss"]["lambda_gan"] = 0.0
        stage1_cfg["loss"]["mse_weight"] = 1.0
        stage1_cfg["loss"]["l1_weight"] = 0.0
        stage1_cfg["multiscale_path_loss"] = {
            "enabled": False,
            "scales": [2, 4],
            "weights": [0.6, 0.4],
            "min_valid_ratio": 0.5,
            "loss_weight": 0.0,
        }
        stage1_cfg["prior_residual_path_loss"] = {
            "enabled": True,
            "use_formula_input_channel": True,
            "optimize_residual_only": False,
            "clamp_final_output": True,
            "loss_weight": 0.0,
            "mse_weight": 1.0,
            "l1_weight": 0.0,
            "final_loss_weight_when_residual_only": 0.0,
            "multiscale_loss_weight_when_residual_only": 0.0,
        }
        stage1_cfg["no_data_auxiliary"] = {
            "enabled": False,
            "loss_weight": 0.0,
            "positive_weight": 1.0,
        }
        stage1_cfg["nlos_focus_loss"] = {
            "enabled": False,
            "mode": "rmse",
            "loss_weight": 0.0,
            "selection_alpha": 0.0,
        }

        stage1_name = f"sixtythirdtry63_expert_{spec['expert_id']}"
        stage1_cfg["runtime"]["output_dir"] = f"outputs/{stage1_name}"
        stage1_cfg["runtime"]["resume_checkpoint"] = ""
        stage1_path = stage1_dir / f"{stage1_name}.yaml"
        with stage1_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage1_cfg, handle, sort_keys=False)
        stage1_registry["experts"].append(
            {
                "expert_id": str(spec["expert_id"]),
                "topology_class": topology_class,
                "config": str(stage1_path.relative_to(root)).replace("\\", "/"),
                "checkpoint": f"outputs/{stage1_name}/best_model.pt",
                "model_arch": "pmhhnet",
            }
        )

        stage2_cfg = deepcopy(stage2_base)
        stage2_cfg["data"]["partition_filter"] = {"topology_class": topology_class}
        stage2_cfg["data"].pop("los_sample_filter", None)
        stage2_cfg["data"]["topology_partitioning"] = {
            "density_q1": 0.12,
            "density_q2": 0.28,
            "height_q1": 12.0,
            "height_q2": 28.0,
        }
        stage2_cfg["data"]["image_size"] = 513
        stage2_cfg["data"]["num_workers"] = 4
        stage2_cfg["data"]["val_num_workers"] = 2
        stage2_cfg["data"]["persistent_workers"] = True
        stage2_cfg["data"]["val_persistent_workers"] = False
        stage2_cfg["data"]["prefetch_factor"] = 4
        stage2_cfg["data"]["val_batch_size"] = 2
        stage2_cfg["data"]["path_loss_formula_input"]["enabled"] = True
        stage2_cfg["data"]["path_loss_formula_input"]["include_confidence_channel"] = True
        stage2_cfg["data"]["path_loss_formula_input"]["cache_enabled"] = False
        stage2_cfg["data"]["path_loss_formula_input"]["cache_dir"] = "prior_cache/try63_prior_auto_city_v1"
        stage2_cfg["data"]["path_loss_formula_input"]["cache_version"] = "try63_prior_auto_city_v1"
        stage2_obstruction_cfg = stage2_cfg["data"].setdefault("path_loss_obstruction_features", {})
        stage2_obstruction_cfg["enabled"] = True
        stage2_obstruction_cfg["precomputed_hdf5"] = "../TFGFiftiethTry50/precomputed/obstruction_features_u8.h5"
        stage2_obstruction_cfg["meters_per_pixel"] = 1.0
        stage2_obstruction_cfg["angle_bins"] = 720
        stage2_obstruction_cfg["include_shadow_depth"] = True
        stage2_obstruction_cfg["include_distance_since_los_break"] = True
        stage2_obstruction_cfg["include_max_blocker_height"] = True
        stage2_obstruction_cfg["include_blocker_count"] = True
        stage2_cfg["augmentation"] = {
            "enable": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "rot90_prob": 0.5,
        }
        stage2_cfg["model"]["base_channels"] = model_spec["refiner_channels"]
        stage2_cfg["model"]["out_channels"] = 1
        stage2_cfg["model"]["dropout"] = 0.05
        stage2_cfg["training"]["batch_size"] = 2
        stage2_cfg["training"]["epochs"] = 180
        stage2_cfg["training"]["learning_rate"] = 8e-5
        stage2_cfg["training"]["weight_decay"] = 0.0
        stage2_cfg["training"]["save_every"] = 1
        stage2_cfg["training"]["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
        stage2_cfg["training"]["early_stopping"] = {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.0,
            "rewind_to_best_model": True,
        }
        stage2_cfg["loss"]["mse_weight"] = 1.0
        stage2_cfg["loss"]["l1_weight"] = 0.0
        stage2_cfg["loss"]["multiscale_path_loss"] = {
            "enabled": False,
            "scales": [2, 4],
            "weights": [0.6, 0.4],
            "min_valid_ratio": 0.5,
            "loss_weight": 0.0,
        }
        stage2_cfg["tail_refiner"]["refiner_arch"] = "unet"
        stage2_cfg["tail_refiner"]["refiner_base_channels"] = model_spec["refiner_channels"]
        stage2_cfg["tail_refiner"]["use_gate"] = False
        stage2_cfg["tail_refiner"]["gate_loss_weight"] = 0.0
        stage2_cfg["tail_refiner"]["residual_weight"] = 1.0
        stage2_cfg["tail_refiner"]["final_weight"] = 1.0
        stage2_cfg["tail_refiner"]["stage1_config"] = str(stage1_path.relative_to(root)).replace("\\", "/")
        stage2_cfg["tail_refiner"]["stage1_checkpoint"] = f"outputs/{stage1_name}/best_model.pt"
        stage2_cfg["tail_refiner"]["oversample"] = {
            "enabled": False,
            "threshold_db": 6.0,
            "temperature_db": 2.5,
            "alpha": 1.0,
            "nlos_boost": 0.0,
            "antenna_boost": 0.0,
        }
        stage2_cfg["tail_refiner"]["tail_focus"] = {
            "enabled": False,
            "threshold_db": 6.0,
            "temperature_db": 2.5,
            "alpha": 1.0,
            "nlos_boost": 0.0,
            "antenna_boost": 0.0,
            "max_weight": 1.0,
        }
        stage2_cfg["tail_refiner"]["high_frequency_loss"] = {
            "enabled": True,
            "laplacian_weight": 0.04,
            "gradient_weight": 0.015,
        }
        stage2_cfg["tail_refiner"]["regime_reweighting"] = {
            "enabled": False,
            "los_weight": 1.0,
            "nlos_weight": 1.0,
            "low_antenna_boost": 0.0,
            "city_type_weights": {
                "open_lowrise": 1.0,
                "mixed_midrise": 1.0,
                "dense_highrise": 1.0,
            },
            "default_city_weight": 1.0,
            "min_weight": 1.0,
            "max_weight": 1.0,
        }

        stage2_name = f"sixtythirdtry63_tail_refiner_{spec['expert_id']}"
        stage2_cfg["runtime"]["output_dir"] = f"outputs/{stage2_name}"
        stage2_cfg["runtime"]["resume_checkpoint"] = ""
        stage2_path = stage2_dir / f"{stage2_name}.yaml"
        with stage2_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage2_cfg, handle, sort_keys=False)
        stage2_registry["experts"].append(
            {
                "expert_id": str(spec["expert_id"]),
                "topology_class": topology_class,
                "config": str(stage2_path.relative_to(root)).replace("\\", "/"),
                "checkpoint": f"outputs/{stage2_name}/best_tail_refiner.pt",
                "stage1_output": f"outputs/{stage1_name}",
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
            "output_dir": "outputs/sixtythirdtry63_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    classifier_path = classifier_dir / "sixtythirdtry63_topology_classifier.yaml"
    with classifier_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(classifier_cfg, handle, sort_keys=False)

    with (stage1_dir / "sixtythirdtry63_expert_registry.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(stage1_registry, handle, sort_keys=False)
    with (stage2_dir / "sixtythirdtry63_tail_refiner_registry.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(stage2_registry, handle, sort_keys=False)


if __name__ == "__main__":
    main()
