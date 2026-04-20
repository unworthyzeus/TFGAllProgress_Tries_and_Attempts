"""Try 77 — minimal runtime-config preparer.

Applies:
    - Optional HDF5 path override (cluster path).
    - Optional resume checkpoint override on runtime.resume_checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

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


def resolve_runtime_paths(cfg: Dict[str, Any], config_root: Path) -> Dict[str, Any]:
    runtime_cfg = dict(cfg.get("runtime", {}))

    output_dir = runtime_cfg.get("output_dir")
    if output_dir:
        output_path = Path(str(output_dir))
        if not output_path.is_absolute():
            runtime_cfg["output_dir"] = str((config_root / output_path).resolve())

    resume_checkpoint = runtime_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        resume_path = Path(str(resume_checkpoint))
        if not resume_path.is_absolute():
            runtime_cfg["resume_checkpoint"] = str((config_root / resume_path).resolve())

    cfg["runtime"] = runtime_cfg
    return cfg


def apply_overrides(
    cfg: Dict[str, Any],
    hdf5_path: Optional[str],
    resume_checkpoint: Optional[str],
) -> Dict[str, Any]:
    if hdf5_path:
        data_cfg = dict(cfg.get("data", {}))
        data_cfg["hdf5_path"] = hdf5_path
        cfg["data"] = data_cfg

    if resume_checkpoint:
        runtime_cfg = dict(cfg.get("runtime", {}))
        runtime_cfg["resume_checkpoint"] = resume_checkpoint
        cfg["runtime"] = runtime_cfg

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a runtime config for Try 77.")
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--hdf5-path", default="", help="Override data.hdf5_path.")
    parser.add_argument("--resume-checkpoint", default="", help="Override runtime.resume_checkpoint.")
    args = parser.parse_args()

    in_path = Path(args.input_config)
    out_path = Path(args.output_config)

    cfg = load_yaml(in_path)
    cfg = resolve_runtime_paths(cfg, config_root=in_path.parent)
    cfg = apply_overrides(cfg, args.hdf5_path or None, args.resume_checkpoint or None)
    save_yaml(out_path, cfg)

    print(f"input_config={in_path}")
    print(f"output_config={out_path}")
    print(f"hdf5_path={cfg.get('data', {}).get('hdf5_path')}")
    print(f"output_dir={cfg.get('runtime', {}).get('output_dir')}")
    print(f"resume_checkpoint={cfg.get('runtime', {}).get('resume_checkpoint')}")


if __name__ == "__main__":
    main()
