"""Generate Try 67 configs: 3 city-type experts (single-stage 513x513).

See TRY67_DESIGN.md for full rationale. Short version:

  * Partition is by city morphology (open_lowrise / mixed_midrise /
    dense_highrise), classified per-sample from building density + mean
    height against the thresholds stored in
    `prior_calibration/regime_obstruction_train_only_from_try47.json`
    (`city_type_thresholds` block). This generalizes to unseen cities at
    inference time — no city-name lookup required.
  * The three classes match ITU-R P.1411 and 3GPP TR 38.901 environment
    categories (rural/suburban, urban-micro, urban-macro). This replaces
    the 6-way Try66 partition whose thresholds were arbitrary quantiles.
  * Anti-overfitting defaults tightened after Try66 stabilised at ~9.3 dB:
    dropout 0.18, weight_decay 0.03, cutmix_prob 0.45, ReduceLROnPlateau
    at 3e-4 base LR, early-stopping patience 25 with best-model rewind.
  * Per-expert tight inference clamp `clip_min_db` / `clip_max_db` set
    from the 99.99th-percentile range of each city type (CLAMPING_AND_IMPROVEMENTS.md §4).
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
# Three city-morphology experts (city-level routing, ITU / 3GPP motivated)
# ---------------------------------------------------------------------------
EXPERT_SPECS = [
    {
        "expert_id": "open_lowrise",
        "city_type": "open_lowrise",
        "clip_min_db": 60.0,
        "clip_max_db": 125.0,
        "base_channels": 40,
        "hf_channels": 16,
        "nlos_reweight_factor": 4.0,
        "nlos_focus_loss_weight": 0.15,
    },
    {
        "expert_id": "mixed_midrise",
        "city_type": "mixed_midrise",
        "clip_min_db": 58.0,
        "clip_max_db": 135.0,
        "base_channels": 44,
        "hf_channels": 18,
        "nlos_reweight_factor": 5.0,
        "nlos_focus_loss_weight": 0.20,
    },
    {
        "expert_id": "dense_highrise",
        "city_type": "dense_highrise",
        "clip_min_db": 55.0,
        "clip_max_db": 145.0,
        "base_channels": 48,
        "hf_channels": 20,
        "nlos_reweight_factor": 6.0,
        "nlos_focus_loss_weight": 0.25,
    },
]


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


def _apply_common_data(cfg: dict, city_type: str) -> None:
    # NOTE: partition_filter uses city_type (ITU/3GPP-aligned morphology),
    # classified per-sample from density/mean-height against fixed thresholds
    # so the routing generalizes to unseen cities at inference time.
    cfg["data"]["partition_filter"] = {"city_type": city_type}
    cfg["data"].pop("los_sample_filter", None)
    cfg["data"].pop("topology_partitioning", None)
    cfg["data"]["num_workers"] = 6
    cfg["data"]["val_num_workers"] = 4
    cfg["data"]["persistent_workers"] = True
    cfg["data"]["val_persistent_workers"] = True
    cfg["data"]["prefetch_factor"] = 4
    cfg["data"]["exclude_non_ground_targets"] = True
    cfg["data"]["path_loss_saturation_db"] = 180
    cfg["data"]["path_loss_no_data_mask_column"] = "path_loss_no_data_mask"
    cfg["data"]["derive_no_data_from_non_ground"] = True

    formula = cfg["data"].setdefault("path_loss_formula_input", {})
    formula["enabled"] = True
    formula["include_confidence_channel"] = True
    formula["cache_enabled"] = False
    formula["cache_dir"] = "prior_cache/try67_prior_auto_city_v1"
    formula["cache_version"] = "try67_prior_auto_city_v1"
    formula["regime_calibration_json"] = (
        "prior_calibration/regime_obstruction_train_only_from_try47.json"
    )
    prior_freq_ghz = float(formula.get("frequency_ghz", 7.125))

    cfg["data"]["tx_depth_map_channel"] = True
    cfg["data"]["elevation_angle_map_channel"] = True
    cfg["data"]["building_mask_channel"] = True

    # SOA #1 — knife-edge diffraction channel (ITU-R P.526 Bullington single-edge).
    # Computed per-sample on-the-fly from the topology heightmap in data_utils.
    # See knife_edge.py and TRY67_DESIGN.md §5 #1.
    # Carrier matches formula prior so both channels describe one scene (was 3.5 vs 7.125).
    cfg["data"]["knife_edge_channel"] = {
        "enabled": True,
        "frequency_ghz": prior_freq_ghz,
        "meters_per_pixel": 1.0,
        "rx_height_m": 1.5,
        "num_ray_samples": 48,
        "scale_db": 40.0,
    }

    obs = cfg["data"].setdefault("path_loss_obstruction_features", {})
    obs["enabled"] = False

    cfg["augmentation"] = {
        "enable": True,
        "hflip_prob": 0.5,
        "vflip_prob": 0.5,
        "rot90_prob": 0.5,
    }


def _build_expert(cfg: dict, spec: dict) -> dict:
    _apply_common_data(cfg, spec["city_type"])

    cfg["data"]["image_size"] = 513
    cfg["data"]["val_batch_size"] = 2

    t = cfg["training"]
    t["batch_size"] = 2
    t["gradient_accumulation_steps"] = 8
    t["epochs"] = 800
    t["optimizer"] = "adamw"
    # TRY67_DESIGN: 3e-4 peak LR (was 4e-4 in Try 66; avoids re-overfitting after rewind).
    t["learning_rate"] = 3.0e-4
    # TRY67_DESIGN: 0.03 (was 0.015 in Try 66). Do not use 0.1 here — that was a typo
    # in an earlier generator revision and shrinks weights ~3× too fast with manual WD.
    t["weight_decay"] = 3.0e-2
    t["ema_decay"] = 0.995

    # Replaces cosine_warm_restarts: LR-resets at cycle boundaries caused the
    # post-best overfitting observed in Try66 open_sparse_lowrise.
    t["lr_scheduler"] = "reduce_on_plateau"
    t["lr_scheduler_factor"] = 0.5
    t["lr_scheduler_patience"] = 8
    t["lr_scheduler_min_lr"] = 1.0e-6
    t["lr_warmup_optimizer_steps"] = 500
    t["lr_warmup_start_factor"] = 0.1

    t["generator_objective"] = "full_map_rmse_only"
    t["selection_metrics"] = {"path_loss.rmse_physical": 1.0}
    t["save_every"] = 5
    t["run_final_test_after_training"] = True
    t["save_validation_json_each_epoch"] = True

    t["early_stopping"] = {
        "enabled": True,
        "patience": 25,
        "min_delta": 0.0,
        "rewind_to_best_model": True,
    }

    t["nlos_reweight_factor"] = float(spec["nlos_reweight_factor"])

    # CutMix raised 0.25 -> 0.45: strongest single regulariser in limited-data
    # dense prediction (Zhang et al., MICCAI 2024). Compensates for the smaller
    # per-expert sample pool when switching to city-level routing.
    t["cutmix_prob"] = 0.45
    t["cutmix_alpha"] = 1.0

    t["regime_reweighting"] = {"enabled": False}
    t["clip_grad_norm"] = 1.0
    t["amp"] = True

    m = cfg["model"]
    m["arch"] = "pmhhnet"
    # Input channels: topology + los + distance_map + formula_input + prior_confidence
    #                + tx_depth_map + elevation_angle + building_mask + knife_edge
    #                = 9 channels (Try51 base had 8; knife_edge is new in Try67)
    m["in_channels"] = 9
    m["base_channels"] = int(spec["base_channels"])
    m["hf_channels"] = int(spec["hf_channels"])
    m["disc_base_channels"] = 0
    # Single output channel: dual LoS/NLoS head was disabled because in
    # open_lowrise environments NLoS pixels are sparse and the NLoS head
    # receives too little gradient to learn — producing 57+ dB NLoS RMSE.
    # NLoS supervision is handled by nlos_focus_loss instead.
    m["out_channels"] = 1
    # Higher dropout (0.12 -> 0.20): primary cheap regulariser after the
    # Try66 9.3 dB plateau.
    m["dropout"] = 0.20
    m["gradient_checkpointing"] = False
    m["use_scalar_film"] = True
    m["use_scalar_channels"] = True
    m["use_se_attention"] = True
    m["se_reduction"] = 4
    m["norm_type"] = "group"

    lo = cfg["loss"]
    lo["lambda_recon"] = 0.0
    lo["lambda_gan"] = 0.0
    # Pure MSE: Huber main-loss route was masking overfitting (tolerant of
    # large errors > delta). Going back to MSE for cleaner gradient signal
    # on the hard NLoS tail that dominates the residual error.
    lo["loss_type"] = "mse"
    lo["mse_weight"] = 0.0
    lo["l1_weight"] = 0.0

    cfg["multiscale_path_loss"] = {
        "enabled": True,
        "scales": [2, 4],
        "weights": [0.6, 0.4],
        "min_valid_ratio": 0.5,
        "loss_weight": 0.3,
    }

    cfg["prior_residual_path_loss"] = {
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

    cfg["no_data_auxiliary"] = {
        "enabled": False,
        "loss_weight": 0.0,
        "positive_weight": 1.0,
    }

    # Dedicated NLoS tail supervision: keep the NLoS branch from collapsing
    # into the LoS-only objective by adding a conservative NLoS RMSE term.
    cfg["nlos_focus_loss"] = {
        "enabled": True,
        "mode": "rmse",
        "loss_weight": float(spec["nlos_focus_loss_weight"]),
    }

    # Soft PINN regulariser (§5 #2). Penalises |∇² pred| on LoS + valid
    # support; motivated by the Helmholtz residual used in ReVeal
    # (arXiv:2502.19646). Low weight to avoid over-smoothing edges.
    cfg["pde_residual_loss"] = {
        "enabled": True,
        "loss_weight": 0.01,
    }

    cfg["dual_los_nlos_head"] = {"enabled": False}

    cfg["test_time_augmentation"] = {
        "enabled": True,
        "transforms": "d4",
        "use_in_validation": False,
        "use_in_final_test": True,
    }

    # Target metadata with per-expert tight physical clamp.
    tmeta = cfg.setdefault("target_metadata", {}).setdefault("path_loss", {})
    tmeta["scale"] = 180.0
    tmeta["offset"] = 0.0
    tmeta["unit"] = "dB"
    tmeta["clip_min"] = 0.0
    tmeta["clip_max"] = 180.0
    tmeta["clip_min_db"] = float(spec["clip_min_db"])
    tmeta["clip_max_db"] = float(spec["clip_max_db"])
    tmeta["predict_linear"] = False

    return cfg


def main() -> None:
    root = ROOT
    base_cfg = _base_config(root)

    expert_dir = root / "experiments" / "sixtyseventh_try67_experts"
    cl_dir = root / "experiments" / "sixtyseventh_try67_classifier"
    expert_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)

    _cleanup(expert_dir, "try67_expert_*.yaml")
    _cleanup(expert_dir, "try67_expert_registry.yaml")

    registry = {"experts": []}

    for spec in EXPERT_SPECS:
        cfg = _build_expert(deepcopy(base_cfg), spec)
        name = f"try67_expert_{spec['expert_id']}"
        cfg["runtime"]["output_dir"] = f"outputs/{name}"
        cfg["runtime"]["resume_checkpoint"] = ""

        path = expert_dir / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        registry["experts"].append({
            "expert_id": spec["expert_id"],
            "city_type": spec["city_type"],
            "config": str(path.relative_to(root)).replace("\\", "/"),
            "checkpoint": f"outputs/{name}/best_model.pt",
            "model_arch": "pmhhnet",
            "clip_min_db": spec["clip_min_db"],
            "clip_max_db": spec["clip_max_db"],
        })

    with (expert_dir / "try67_expert_registry.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(registry, f, sort_keys=False)

    print(f"Generated {len(EXPERT_SPECS)} city-type expert configs in {expert_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
