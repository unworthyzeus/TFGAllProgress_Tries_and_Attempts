from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils import TRY54_TOPOLOGY_CLASSES

TOPOLOGY_MODEL_SPECS = {
    "open_sparse_lowrise": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
    "open_sparse_vertical": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
    "mixed_compact_lowrise": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
    "mixed_compact_midrise": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
    "dense_block_midrise": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
    "dense_block_highrise": {"base_channels": 10, "hf_channels": 8, "scalar_hidden_dim": 24},
}


def main() -> None:
    root = ROOT
    base_stage1 = (
        root.parent
        / "TFGFiftyFirstTry51"
        / "experiments"
        / "fiftyfirsttry51_pmnet_prior_gan_fastbatch"
        / "fiftyfirsttry51_pmnet_prior_stage1_widen112_initial_literature.yaml"
    )
    out_dir = root / "experiments" / "fiftyfifthtry55_partitioned_stage1"
    classifier_dir = root / "experiments" / "fiftyfifthtry55_topology_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir.mkdir(parents=True, exist_ok=True)

    for stale in out_dir.glob("fiftyfifthtry55_expert_*.yaml"):
        stale.unlink()

    with base_stage1.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    registry = {"experts": []}
    for topology_class in TRY54_TOPOLOGY_CLASSES:
        model_spec = TOPOLOGY_MODEL_SPECS[topology_class]
        cfg = deepcopy(base_cfg)
        cfg["data"]["partition_filter"] = {"topology_class": topology_class}
        cfg["data"]["topology_partitioning"] = {
            "density_q1": 0.12,
            "density_q2": 0.28,
            "height_q1": 12.0,
            "height_q2": 28.0,
        }
        cfg["data"]["image_size"] = 513
        cfg["data"]["num_workers"] = 4
        cfg["data"]["val_num_workers"] = 1
        cfg["data"]["persistent_workers"] = True
        cfg["data"]["val_persistent_workers"] = False
        cfg["data"]["prefetch_factor"] = 2
        cfg["data"]["exclude_non_ground_targets"] = True
        cfg["data"]["path_loss_saturation_db"] = 180
        cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
        cfg["data"]["derive_no_data_from_non_ground"] = True
        cfg["data"]["path_loss_formula_input"]["cache_enabled"] = True
        cfg["data"]["path_loss_formula_input"]["cache_dir"] = "prior_cache/try55_prior_auto_city_v1"
        cfg["data"]["path_loss_formula_input"]["cache_version"] = "try55_prior_auto_city_v1"
        cfg["data"]["path_loss_formula_input"]["cache_dtype"] = "uint8"
        cfg["training"]["epochs"] = 1000
        cfg["training"]["batch_size"] = 6
        cfg["training"]["optimizer"] = "adamw"
        cfg["training"]["learning_rate"] = 8e-4
        cfg["training"]["weight_decay"] = 0.075
        cfg["training"]["ema_decay"] = 1.0
        cfg["training"]["lr_scheduler"] = "none"
        cfg["training"]["lr_scheduler_factor"] = 1.0
        cfg["training"]["lr_scheduler_patience"] = 9999
        cfg["training"]["lr_scheduler_min_lr"] = 0.0
        cfg["training"]["save_every"] = 1
        cfg["training"]["generator_objective"] = "full_map_rmse_only"
        cfg["training"]["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
        cfg["training"]["early_stopping"] = {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.0,
            "rewind_to_best_model": True,
        }
        cfg["training"]["regime_reweighting"]["enabled"] = False
        cfg["model"]["disc_base_channels"] = 0
        cfg["model"]["arch"] = "pmhhnet"
        cfg["model"]["out_channels"] = 2
        cfg["model"]["base_channels"] = model_spec["base_channels"]
        cfg["model"]["hf_channels"] = model_spec["hf_channels"]
        cfg["model"]["scalar_hidden_dim"] = model_spec["scalar_hidden_dim"]
        cfg["model"]["encoder_blocks"] = [1, 2, 2, 2]
        cfg["model"]["context_dilations"] = [1, 2, 4, 8]
        cfg["model"]["dropout"] = 0.08
        cfg["model"]["use_scalar_channels"] = True
        cfg["model"]["use_scalar_film"] = True
        cfg["model"]["gradient_checkpointing"] = False
        cfg["loss"]["lambda_gan"] = 0.0
        cfg["loss"]["mse_weight"] = 1.0
        cfg["loss"]["l1_weight"] = 0.0
        cfg["multiscale_path_loss"]["enabled"] = False
        cfg["multiscale_path_loss"]["loss_weight"] = 0.0
        cfg["prior_residual_path_loss"]["loss_weight"] = 0.0
        cfg["prior_residual_path_loss"]["mse_weight"] = 1.0
        cfg["prior_residual_path_loss"]["l1_weight"] = 0.0
        cfg["prior_residual_path_loss"]["final_loss_weight_when_residual_only"] = 0.0
        cfg["prior_residual_path_loss"]["multiscale_loss_weight_when_residual_only"] = 0.0
        cfg["no_data_auxiliary"] = {
            "enabled": True,
            "loss_weight": 0.18,
            "positive_weight": 1.8,
            "description": "Auxiliary no-data segmentation head trained on pixels currently masked out of path-loss regression.",
        }

        expert_name = f"fiftyfifthtry55_expert_{topology_class}"
        cfg["runtime"]["output_dir"] = f"outputs/{expert_name}"
        cfg["runtime"]["resume_checkpoint"] = ""

        out_path = out_dir / f"{expert_name}.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        registry["experts"].append(
            {
                "topology_class": topology_class,
                "config": str(out_path.relative_to(root)).replace("\\", "/"),
                "checkpoint": f"outputs/{expert_name}/best_model.pt",
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
            "output_dir": "outputs/fiftyfifthtry55_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    classifier_path = classifier_dir / "fiftyfifthtry55_topology_classifier.yaml"
    with classifier_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(classifier_cfg, handle, sort_keys=False)

    registry_path = out_dir / "fiftyfifthtry55_expert_registry.yaml"
    with registry_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False)


if __name__ == "__main__":
    main()
