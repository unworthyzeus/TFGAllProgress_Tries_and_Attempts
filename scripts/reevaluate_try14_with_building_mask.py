from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def denormalize(values: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    scale = float(metadata.get("scale", 1.0))
    offset = float(metadata.get("offset", 0.0))
    return values * scale + offset


def main() -> None:
    parser = argparse.ArgumentParser(description="Reevaluate a cGAN/U-Net checkpoint using the current building mask.")
    parser.add_argument(
        "--config",
        default=(
            r"C:\TFG\TFGpractice\TFGFourteenthTry14\experiments\fourteenthtry14_film"
            r"\fourteenthtry14_nlos_film.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            r"C:\TFG\TFGpractice\cluster_outputs\TFGFourteenthTry14\cgan_unet_hdf5_pathloss_hybrid_cuda_max112_"
            r"blend_db_tinygan_batchnorm_lowresdisc_fourteenthtry14_nlos_film_ddp2_t14_nlos_film\best_cgan.pt"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=r"C:\TFG\TFGpractice\Datasets\CKM_Dataset_180326_antenna_height.h5",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exclude-non-ground", action="store_true")
    parser.add_argument(
        "--out-json",
        default=r"C:\TFG\TFGpractice\analysis\try14_nlos_buildingmask_reeval.json",
    )
    parser.add_argument(
        "--try-root",
        default=None,
        help="Root folder of the try to reevaluate. If omitted, inferred from --config.",
    )
    parser.add_argument(
        "--mask-data-root",
        default=r"C:\TFG\TFGpractice\TFGThirtyThirdTry33",
        help="Try folder providing the modern masked data_utils.py implementation.",
    )
    args = parser.parse_args()

    repo_root = Path(r"C:\TFG\TFGpractice")
    config_path = Path(args.config).resolve()
    inferred_try_root = config_path.parents[2] if len(config_path.parents) >= 3 else config_path.parent
    try_root = Path(args.try_root).resolve() if args.try_root else inferred_try_root
    mask_root = Path(args.mask_data_root).resolve()
    if str(mask_root) not in sys.path:
        sys.path.insert(0, str(mask_root))
    if str(try_root) not in sys.path:
        sys.path.insert(0, str(try_root))

    config_utils = load_module(f"{try_root.name.lower()}_config_utils", try_root / "config_utils.py")
    model_cgan = load_module(f"{try_root.name.lower()}_model_cgan", try_root / "model_cgan.py")
    mask_data_utils = load_module("t33_data_utils", mask_root / "data_utils.py")

    cfg = config_utils.load_config(args.config)
    config_utils.anchor_data_paths_to_config_file(cfg, args.config)
    cfg["data"]["hdf5_path"] = args.dataset
    cfg["data"]["exclude_non_ground_targets"] = bool(args.exclude_non_ground)
    cfg["data"]["non_ground_threshold"] = 0.0
    cfg["data"]["num_workers"] = int(args.num_workers)

    splits = mask_data_utils.build_dataset_splits_from_config(cfg)
    if args.split not in splits:
        raise ValueError(f"Split {args.split!r} not available.")
    dataset = splits[args.split]

    in_channels = mask_data_utils.compute_input_channels(cfg)
    scalar_cond_dim = (
        mask_data_utils.compute_scalar_cond_dim(cfg)
        if mask_data_utils.return_scalar_cond_from_config(cfg)
        else 0
    )
    generator = model_cgan.UNetGenerator(
        in_channels=in_channels,
        out_channels=int(cfg["model"]["out_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        gradient_checkpointing=False,
        path_loss_hybrid=bool(cfg.get("path_loss_hybrid", {}).get("enabled", False)),
        norm_type=str(cfg["model"].get("norm_type", "batch")),
        scalar_cond_dim=scalar_cond_dim,
        scalar_film_hidden=int(cfg["model"].get("scalar_film_hidden", 128)),
        upsample_mode=str(cfg["model"].get("upsample_mode", "transpose")),
    )

    device = config_utils.resolve_device(args.device)
    state = config_utils.load_torch_checkpoint(args.checkpoint, device)
    generator.load_state_dict(state["generator"] if "generator" in state else state, strict=True)
    generator.to(device)
    generator.eval()

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(getattr(device, "type", None) == "cuda"),
    )

    meta = dict(cfg["target_metadata"]["path_loss"])
    los_num = 0.0
    nlos_num = 0.0
    total_num = 0.0
    sse = 0.0
    sae = 0.0
    los_sse = 0.0
    nlos_sse = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="reeval", leave=False):
            if len(batch) == 4:
                x, y, m, scalar_cond = batch
                scalar_cond = scalar_cond.to(device)
            else:
                x, y, m = batch
                scalar_cond = None
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            pred = mask_data_utils.forward_cgan_generator(generator, x, scalar_cond)
            pred = pred[:, :1]
            target = y[:, :1]
            mask = m[:, :1] > 0.0
            pred_phys = denormalize(pred, meta)
            target_phys = denormalize(target, meta)
            diff = pred_phys - target_phys
            valid = diff[mask]
            if valid.numel() == 0:
                continue
            sse += float(torch.sum(valid ** 2).item())
            sae += float(torch.sum(torch.abs(valid)).item())
            total_num += float(valid.numel())

            if x.size(1) > 1:
                los_mask = x[:, 1:2] > 0.5
                los_valid = mask & los_mask
                nlos_valid = mask & (~los_mask)
                los_vals = diff[los_valid]
                nlos_vals = diff[nlos_valid]
                if los_vals.numel() > 0:
                    los_sse += float(torch.sum(los_vals ** 2).item())
                    los_num += float(los_vals.numel())
                if nlos_vals.numel() > 0:
                    nlos_sse += float(torch.sum(nlos_vals ** 2).item())
                    nlos_num += float(nlos_vals.numel())

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "try_root": str(try_root),
        "mask_data_root": str(mask_root),
        "dataset": str(args.dataset),
        "split": args.split,
        "mask_rule": (
            "only pixels with topology == 0 contribute to target mask and error"
            if args.exclude_non_ground
            else "legacy mask (no explicit topology exclusion)"
        ),
        "path_loss": {
            "mse_physical": sse / max(total_num, 1.0),
            "rmse_physical": math.sqrt(sse / max(total_num, 1.0)),
            "mae_physical": sae / max(total_num, 1.0),
            "unit": str(meta.get("unit", "dB")),
            "valid_pixels": int(total_num),
        },
        "_regimes": {
            "path_loss__los__LoS": {
                "rmse_physical": math.sqrt(los_sse / max(los_num, 1.0)) if los_num > 0 else float("nan"),
                "valid_pixels": int(los_num),
                "unit": str(meta.get("unit", "dB")),
            },
            "path_loss__los__NLoS": {
                "rmse_physical": math.sqrt(nlos_sse / max(nlos_num, 1.0)) if nlos_num > 0 else float("nan"),
                "valid_pixels": int(nlos_num),
                "unit": str(meta.get("unit", "dB")),
            },
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
