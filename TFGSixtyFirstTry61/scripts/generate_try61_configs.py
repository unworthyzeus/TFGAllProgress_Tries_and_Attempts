from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRY61_EXPERT_SPECS = [
    {"expert_id": "open_sparse_lowrise", "topology_class": "open_sparse_lowrise"},
    {"expert_id": "open_sparse_vertical_los", "topology_class": "open_sparse_vertical", "los_sample_filter": "los_only"},
    {"expert_id": "open_sparse_vertical_nlos", "topology_class": "open_sparse_vertical", "los_sample_filter": "nlos_only"},
    {"expert_id": "mixed_compact_lowrise", "topology_class": "mixed_compact_lowrise"},
    {"expert_id": "mixed_compact_midrise", "topology_class": "mixed_compact_midrise"},
    {"expert_id": "dense_block_midrise", "topology_class": "dense_block_midrise"},
    {"expert_id": "dense_block_highrise", "topology_class": "dense_block_highrise"},
]

TOPOLOGY_MODEL_SPECS = {
    "open_sparse_lowrise": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
    "open_sparse_vertical": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
    "mixed_compact_lowrise": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
    "mixed_compact_midrise": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
    "dense_block_midrise": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
    "dense_block_highrise": {"base_channels": 12, "hf_channels": 10, "scalar_hidden_dim": 28},
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
    out_dir = root / "experiments" / "sixtyfirsttry61_partitioned_stage1"
    classifier_dir = root / "experiments" / "sixtyfirsttry61_topology_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir.mkdir(parents=True, exist_ok=True)

    for stale in out_dir.glob("sixtyfirsttry61_expert_*.yaml"):
        try:
            stale.unlink()
        except PermissionError:
            pass

    with base_stage1.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)

    registry = {"experts": []}
    for spec in TRY61_EXPERT_SPECS:
        topology_class = str(spec["topology_class"])
        model_spec = TOPOLOGY_MODEL_SPECS[topology_class]
        cfg = deepcopy(base_cfg)
        cfg["data"]["partition_filter"] = {"topology_class": topology_class}
        if spec.get("los_sample_filter"):
            cfg["data"]["los_sample_filter"] = str(spec["los_sample_filter"])
        else:
            cfg["data"].pop("los_sample_filter", None)
        cfg["data"]["topology_partitioning"] = {
            "density_q1": 0.12,
            "density_q2": 0.28,
            "height_q1": 12.0,
            "height_q2": 28.0,
        }
        cfg["data"]["image_size"] = 513
        cfg["data"]["num_workers"] = 4
        cfg["data"]["val_num_workers"] = 2
        cfg["data"]["persistent_workers"] = True
        cfg["data"]["val_persistent_workers"] = False
        cfg["data"]["prefetch_factor"] = 2
        cfg["data"]["exclude_non_ground_targets"] = True
        cfg["data"]["path_loss_saturation_db"] = 180
        cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
        cfg["data"]["derive_no_data_from_non_ground"] = True
        formula_cfg = cfg["data"].setdefault("path_loss_formula_input", {})
        formula_cfg["enabled"] = False
        formula_cfg["include_confidence_channel"] = False
        formula_cfg["cache_enabled"] = False
        formula_cfg["cache_dir"] = ""
        formula_cfg["cache_version"] = "try61_no_prior"
        cfg["training"]["epochs"] = 1000
        cfg["training"]["batch_size"] = 5
        cfg["training"]["optimizer"] = "adamw"
        cfg["training"]["learning_rate"] = 8e-4
        cfg["training"]["weight_decay"] = 0.06
        cfg["training"]["ema_decay"] = 1.0
        cfg["training"]["lr_scheduler"] = "none"
        cfg["training"]["lr_scheduler_factor"] = 1.0
        cfg["training"]["lr_scheduler_patience"] = 9999
        cfg["training"]["lr_scheduler_min_lr"] = 0.0
        cfg["training"]["save_every"] = 1
        cfg["training"]["generator_objective"] = "legacy"
        cfg["training"]["selection_metrics"] = {"selection_proxy.composite_nlos_weighted_rmse": 1.0}
        cfg["training"]["selection_nlos_alpha"] = 0.25
        cfg["training"]["early_stopping"] = {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.0,
            "rewind_to_best_model": True,
        }
        cfg["training"]["regime_reweighting"]["enabled"] = True
        cfg["training"]["regime_reweighting"]["los_weight"] = 0.5
        cfg["training"]["regime_reweighting"]["nlos_weight"] = 4.0
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
        cfg["loss"]["lambda_recon"] = 1.0
        cfg["loss"]["lambda_gan"] = 0.0
        cfg["loss"]["mse_weight"] = 1.0
        cfg["loss"]["l1_weight"] = 0.1
        cfg["multiscale_path_loss"]["enabled"] = True
        cfg["multiscale_path_loss"]["loss_weight"] = 0.2
        cfg["prior_residual_path_loss"]["optimize_residual_only"] = False
        cfg["prior_residual_path_loss"]["loss_weight"] = 0.0
        cfg["prior_residual_path_loss"]["mse_weight"] = 1.0
        cfg["prior_residual_path_loss"]["l1_weight"] = 0.1
        cfg["prior_residual_path_loss"]["final_loss_weight_when_residual_only"] = 0.0
        cfg["prior_residual_path_loss"]["multiscale_loss_weight_when_residual_only"] = 0.0
        cfg["no_data_auxiliary"] = {
            "enabled": True,
            "loss_weight": 0.05,
            "positive_weight": 1.8,
            "description": "Auxiliary no-data segmentation head trained on pixels currently masked out of path-loss regression.",
        }
        cfg["nlos_focus_loss"] = {
            "enabled": True,
            "mode": "rmse",
            "loss_weight": 1.0,
            "selection_alpha": 0.25,
        }

        expert_name = f"sixtyfirsttry61_expert_{spec['expert_id']}"
        cfg["runtime"]["output_dir"] = f"outputs/{expert_name}"
        cfg["runtime"]["resume_checkpoint"] = ""

        out_path = out_dir / f"{expert_name}.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

        registry["experts"].append(
            {
                "expert_id": str(spec["expert_id"]),
                "topology_class": topology_class,
                "los_sample_filter": spec.get("los_sample_filter"),
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
            "output_dir": "outputs/sixtyfirsttry61_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    classifier_path = classifier_dir / "sixtyfirsttry61_topology_classifier.yaml"
    with classifier_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(classifier_cfg, handle, sort_keys=False)

    registry_path = out_dir / "sixtyfirsttry61_expert_registry.yaml"
    with registry_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False)


if __name__ == "__main__":
    main()
