from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def anchor_data_paths_to_config_file(cfg: Dict[str, Any], config_path: str) -> None:
    """
    Resolve relative data.hdf5_path and data.scalar_table_csv against the try-folder root
    (parent of configs/), so training works no matter the process cwd.
    """
    cfg_p = Path(config_path).resolve()
    if cfg_p.parent.name == "configs":
        root = cfg_p.parent.parent
    else:
        root = cfg_p.parent
    data = cfg.get("data")
    if not isinstance(data, dict):
        return
    for key in ("hdf5_path", "scalar_table_csv"):
        val = data.get(key)
        if not val or not isinstance(val, str):
            continue
        p = Path(val)
        if p.is_absolute():
            continue
        resolved = (root / p).resolve()
        data[key] = str(resolved)


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with cfg_path.open('r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def is_cuda_device(device: Any) -> bool:
    if isinstance(device, str):
        return device == 'cuda'
    try:
        return getattr(device, 'type', None) == 'cuda'
    except Exception:
        return False


def resolve_device(device_cfg: str) -> Any:
    device_name = str(device_cfg).lower()

    try:
        import torch
    except Exception:
        return 'cpu'

    if device_name == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            return 'cpu'

    if device_name in {'cuda', 'cpu'}:
        return device_name

    if device_name in {'directml', 'dml', 'amd'}:
        try:
            import torch_directml
            return torch_directml.device()
        except Exception as exc:
            raise RuntimeError(
                "DirectML device requested but torch-directml is not installed. Install it and retry."
            ) from exc

    return device_cfg


def resolve_checkpoint_map_location(device: Any) -> Any:
    if is_cuda_device(device):
        return device
    return 'cpu'


def load_torch_checkpoint(path: str | Path, device: Any) -> Any:
    import torch

    map_location = resolve_checkpoint_map_location(device)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _move_state_value_to_device(value: Any, device: Any) -> Any:
    try:
        import torch
    except Exception:
        return value

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_state_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_value_to_device(item, device) for item in value)
    return value


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    for state_id, state_value in list(optimizer.state.items()):
        optimizer.state[state_id] = _move_state_value_to_device(state_value, device)
