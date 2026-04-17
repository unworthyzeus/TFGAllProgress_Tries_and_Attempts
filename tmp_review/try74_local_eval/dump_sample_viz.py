"""Dump a small sheet of path_loss / los / topology for a few samples,
so we can eyeball the 2-ray ring pattern around the antenna."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def _norm_pl(pl: np.ndarray) -> np.ndarray:
    mn = float(np.nanmin(pl[pl > 0])) if np.any(pl > 0) else 0.0
    mx = float(np.nanmax(pl))
    if mx - mn < 1e-6:
        return np.zeros_like(pl, dtype=np.uint8)
    arr = (pl - mn) / (mx - mn)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def dump_sample(h, city: str, sample: str, out_dir: Path) -> None:
    g = h[city][sample]
    pl = np.asarray(g["path_loss"][...], dtype=np.float32)
    topo = np.asarray(g["topology_map"][...], dtype=np.float32)
    los = np.asarray(g["los_mask"][...], dtype=np.float32)
    try:
        uav_h = float(np.asarray(g["uav_height"][...]).reshape(-1)[0])
    except Exception:
        uav_h = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{city}_{sample}"

    Image.fromarray(_norm_pl(pl), mode="L").save(out_dir / f"{stem}__pathloss.png")

    topo_u8 = np.clip(topo, 0, 255).astype(np.uint8)
    Image.fromarray(topo_u8, mode="L").save(out_dir / f"{stem}__topology.png")

    los_u8 = (los > 0.5).astype(np.uint8) * 255
    Image.fromarray(los_u8, mode="L").save(out_dir / f"{stem}__los.png")

    # Radial profile: azimuthally-averaged path_loss vs distance from center.
    h_, w = pl.shape
    cy, cx = (h_ - 1) / 2.0, (w - 1) / 2.0
    y, x = np.ogrid[:h_, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    max_r = int(r.max())
    radial_sum = np.bincount(r.ravel(), weights=pl.ravel().astype(np.float64), minlength=max_r + 1)
    radial_cnt = np.bincount(r.ravel(), minlength=max_r + 1)
    radial_mean = np.divide(radial_sum, np.maximum(radial_cnt, 1))

    # Dump CSV: r_px, mean_pl_dB, count
    with (out_dir / f"{stem}__radial.csv").open("w") as f:
        f.write("r_px,r_m,mean_path_loss_dB,count\n")
        for i, (mean, cnt) in enumerate(zip(radial_mean, radial_cnt)):
            f.write(f"{i},{i},{mean:.3f},{int(cnt)}\n")

    print(f"{city}/{sample}: uav_h={uav_h:.1f}m  shape={pl.shape}  pl range=[{pl.min():.0f},{pl.max():.0f}] dB  "
          f"los_frac={(los > 0.5).mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default=r"C:/TFG/TFGPractice/Datasets/CKM_Dataset_270326.h5")
    ap.add_argument("--out-dir", default=r"C:/TFG/TFGPractice/tmp_review/try74_local_eval/viz")
    ap.add_argument("--pairs", nargs="+", default=[
        "Dhaka/sample_05022",  # best NLoS in smoke
        "Dhaka/sample_05067",  # worst NLoS in smoke
        "Abidjan/sample_00001",
        "Barcelona/sample_02001",
        "Shanghai/sample_01001",
    ])
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as h:
        for pair in args.pairs:
            city, sample = pair.split("/", 1)
            if city not in h or sample not in h[city]:
                print(f"SKIP (not found): {pair}")
                continue
            dump_sample(h, city, sample, Path(args.out_dir))


if __name__ == "__main__":
    main()
