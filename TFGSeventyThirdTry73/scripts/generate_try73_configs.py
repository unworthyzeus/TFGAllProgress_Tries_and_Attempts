"""Generate Try 73 configs as a stable direct-prediction variant.

We still inherit the mature Try 66/68 training layout, but Try 73 stays
*prior-free*: the network predicts path loss directly instead of learning a
residual over the analytic prior.

Intentional design for this generator:

1. Keep the stable masking / topology-partition / data layout from the mature line.
2. Keep the Try 73 code fixes (stem+HF before FiLM, FiLM-safe no-CutMix).
3. Avoid the failure modes from the first no-prior Try 73 launch:
   - no formula prior / confidence channel,
   - no residual-prior objective,
   - no per-expert physical clamp floor/ceiling,
   - keep the building-mask input channel.
"""

from __future__ import annotations

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

TOPO_PART = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}


def _load_try66_template(root: Path, topology_class: str) -> dict:
    path = (
        root.parent
        / "TFGSixtySixthTry66"
        / "experiments"
        / "sixtysixth_try66_experts"
        / f"try66_expert_{topology_class}.yaml"
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
    topology_class = spec["topology_class"]
    cfg = _load_try66_template(root, topology_class)
    cfg["data"]["partition_filter"] = {"topology_class": topology_class}
    cfg["data"]["topology_partitioning"] = dict(TOPO_PART)
    cfg["data"].pop("los_sample_filter", None)

    formula = cfg["data"].setdefault("path_loss_formula_input", {})
    formula["enabled"] = False
    formula["include_confidence_channel"] = False
    formula["cache_dir"] = "prior_cache/try73_prior_auto_city_v1"
    formula["cache_version"] = "try73_prior_auto_city_v1"

    cfg["data"]["building_mask_channel"] = True
    cfg["data"]["val_batch_size"] = 1

    model_cfg = cfg.setdefault("model", {})
    model_cfg["base_channels"] = 80
    model_cfg["hf_channels"] = 32
    model_cfg["gradient_checkpointing"] = True
    model_cfg["absolute_output_bias_init_db"] = "auto"

    train_cfg = cfg.setdefault("training", {})
    train_cfg["batch_size"] = 1
    train_cfg["learning_rate"] = 5.0e-4
    train_cfg["weight_decay"] = 1.5e-2
    train_cfg["gradient_accumulation_steps"] = 8
    train_cfg["ema_decay"] = 0.95

    prior_cfg = cfg.setdefault("prior_residual_path_loss", {})
    prior_cfg["enabled"] = False
    prior_cfg["absolute_prediction"] = True
    prior_cfg["use_formula_input_channel"] = False
    prior_cfg["optimize_residual_only"] = False
    prior_cfg["loss_weight"] = 0.0
    prior_cfg["clamp_final_output"] = False

    loss_cfg = cfg.setdefault("loss", {})
    loss_cfg["full_map_huber_physical_db"] = True

    target_meta = cfg.setdefault("target_metadata", {}).setdefault("path_loss", {})
    target_meta.pop("clip_min_db", None)
    target_meta.pop("clip_max_db", None)

    cfg.setdefault("corridor_weighting", {})["enabled"] = False
    return cfg


def main() -> None:
    root = ROOT

    expert_dir = root / "experiments" / "seventythird_try73_experts"
    cl_dir = root / "experiments" / "seventythird_try73_classifier"
    expert_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)

    _cleanup(expert_dir, "try73_expert_*.yaml")

    registry = {"experts": []}

    for spec in EXPERT_SPECS:
        tc = spec["topology_class"]
        cfg = _build_expert(root, spec)
        name = f"try73_expert_{spec['expert_id']}"
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
            "output_dir": "outputs/try73_topology_classifier",
            "resume_checkpoint": "",
        },
    }
    cl_path = cl_dir / "try73_topology_classifier.yaml"
    with cl_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cl_cfg, f, sort_keys=False)

    with (expert_dir / "try73_expert_registry.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    print(f"Generated {len(EXPERT_SPECS)} expert configs in {expert_dir}")
    print(f"Generated classifier config in {cl_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
