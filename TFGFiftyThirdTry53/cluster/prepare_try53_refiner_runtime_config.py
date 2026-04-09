from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare runtime config for Try53 stage2/stage3 with overridden outputs/checkpoints.")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-checkpoint", default="")
    parser.add_argument("--teacher-checkpoint", default="")
    parser.add_argument("--val-batch-size", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input_config)
    output_path = Path(args.output_config)

    cfg = load_yaml(input_path)
    runtime_cfg = dict(cfg.get("runtime", {}))
    runtime_cfg["output_dir"] = args.output_dir
    cfg["runtime"] = runtime_cfg

    data_cfg = dict(cfg.get("data", {}))
    data_cfg["val_batch_size"] = int(args.val_batch_size)
    cfg["data"] = data_cfg

    tail_cfg = dict(cfg.get("tail_refiner", {}))
    if args.stage1_checkpoint:
        tail_cfg["stage1_checkpoint"] = args.stage1_checkpoint
    if args.teacher_checkpoint:
        tail_cfg["teacher_checkpoint"] = args.teacher_checkpoint
    cfg["tail_refiner"] = tail_cfg

    save_yaml(output_path, cfg)

    print(f"input_config={input_path}")
    print(f"output_config={output_path}")
    print(f"output_dir={args.output_dir}")
    if args.stage1_checkpoint:
        print(f"stage1_checkpoint={args.stage1_checkpoint}")
    if args.teacher_checkpoint:
        print(f"teacher_checkpoint={args.teacher_checkpoint}")
    print(f"val_batch_size={args.val_batch_size}")


if __name__ == "__main__":
    main()
