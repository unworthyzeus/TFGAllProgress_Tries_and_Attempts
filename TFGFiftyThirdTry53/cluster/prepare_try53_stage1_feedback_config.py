from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_or_none(payload: Dict[str, Any], key_path: list[str]) -> float | None:
    cur: Any = payload
    for key in key_path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return None


def load_latest_metrics(stage3_json: Path, stage2_json: Path) -> tuple[Dict[str, Any], str]:
    if stage3_json.exists():
        return _read_json(stage3_json), "stage3"
    if stage2_json.exists():
        return _read_json(stage2_json), "stage2"
    return {}, "none"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tune_feedback_weights(cfg: Dict[str, Any], metrics: Dict[str, Any], source: str) -> Dict[str, Any]:
    training_cfg = dict(cfg.get("training", {}))
    regime_cfg = dict(training_cfg.get("regime_reweighting", {}))

    current_nlos_weight = float(regime_cfg.get("nlos_weight", 2.5))
    current_low_ant_boost = float(regime_cfg.get("low_antenna_boost", 0.35))

    los_rmse = _metric_or_none(metrics, ["path_loss__los__LoS", "rmse_physical"])
    nlos_rmse = _metric_or_none(metrics, ["path_loss__los__NLoS", "rmse_physical"])

    if los_rmse is None or nlos_rmse is None:
        cfg.setdefault("_try53_feedback", {})
        cfg["_try53_feedback"].update(
            {
                "source": source,
                "applied": False,
                "reason": "missing_los_nlos_rmse",
                "nlos_weight": current_nlos_weight,
                "low_antenna_boost": current_low_ant_boost,
            }
        )
        return cfg

    ratio = nlos_rmse / max(los_rmse, 1.0)
    proposed_nlos_weight = clamp(1.6 + 0.15 * ratio, 2.25, 4.5)
    if nlos_rmse > 30.0:
        proposed_nlos_weight = clamp(proposed_nlos_weight + 0.2, 2.25, 4.5)

    proposed_low_ant_boost = 0.45 if nlos_rmse > 30.0 else 0.35

    regime_cfg["nlos_weight"] = round(proposed_nlos_weight, 3)
    regime_cfg["low_antenna_boost"] = round(proposed_low_ant_boost, 3)

    training_cfg["regime_reweighting"] = regime_cfg
    cfg["training"] = training_cfg

    cfg.setdefault("_try53_feedback", {})
    cfg["_try53_feedback"].update(
        {
            "source": source,
            "applied": True,
            "los_rmse": los_rmse,
            "nlos_rmse": nlos_rmse,
            "nlos_over_los_ratio": ratio,
            "old_nlos_weight": current_nlos_weight,
            "new_nlos_weight": regime_cfg["nlos_weight"],
            "old_low_antenna_boost": current_low_ant_boost,
            "new_low_antenna_boost": regime_cfg["low_antenna_boost"],
        }
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Try53 cyclic feedback tuning to stage1 config.")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--stage3-json",
        default="outputs/fiftythirdtry53_stage3_nlos_global_context_cyclic_4gpu/validate_metrics_tail_refiner_latest.json",
    )
    parser.add_argument(
        "--stage2-json",
        default="outputs/fiftythirdtry53_tail_refiner_stage2_teacher_literature_cyclic_4gpu/validate_metrics_tail_refiner_latest.json",
    )
    args = parser.parse_args()

    input_cfg_path = Path(args.input_config)
    output_cfg_path = Path(args.output_config)
    stage3_json = Path(args.stage3_json)
    stage2_json = Path(args.stage2_json)

    cfg = load_yaml(input_cfg_path)
    metrics, source = load_latest_metrics(stage3_json, stage2_json)
    cfg = tune_feedback_weights(cfg, metrics, source)
    save_yaml(output_cfg_path, cfg)

    feedback = cfg.get("_try53_feedback", {})
    print(json.dumps({"feedback": feedback}, indent=2))


if __name__ == "__main__":
    main()
