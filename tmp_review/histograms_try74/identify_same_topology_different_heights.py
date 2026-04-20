from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


ROOT = Path(r"c:/TFG/TFGPractice")
DEFAULT_CSV = Path(__file__).with_name("histograms.csv")
DEFAULT_CFG = (
    ROOT
    / "TFGSeventyFifthTry75"
    / "experiments"
    / "seventyfifth_try75_experts"
    / "try75_expert_allcity_los.yaml"
)


def resolve_hdf5_path(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    hdf5_rel = str(cfg.get("data", {}).get("hdf5_path", "")).strip()
    if not hdf5_rel:
        raise RuntimeError(f"Could not find data.hdf5_path in {config_path}")

    candidate = (config_path.parent / hdf5_rel).resolve()
    if candidate.exists():
        return candidate

    fallback = ROOT / "datasets" / Path(hdf5_rel).name
    if fallback.exists():
        print(f"[hdf5] config path missing, using datasets fallback: {fallback}")
        return fallback

    raise FileNotFoundError(
        "Unable to resolve HDF5 path from config. Tried: "
        f"{candidate} and {fallback}"
    )


def load_sample_keys_from_csv(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        order: list[str] = []
        seen: set[str] = set()
        for row in reader:
            key = f"{row.get('city', '')}/{row.get('sample', '')}"
            if key not in seen:
                seen.add(key)
                order.append(key)
    return order


def extract_scalar(grp: h5py.Group, name: str, default: float = float("nan")) -> float:
    if name not in grp:
        return default
    return float(np.asarray(grp[name][()]).reshape(()))


def build_topology_groups(
    sample_keys: list[str],
    hdf5_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []

    with h5py.File(hdf5_path, "r") as h5:
        for idx, key in enumerate(sample_keys, start=1):
            city, sample = key.split("/", 1)
            if city not in h5 or sample not in h5[city]:
                missing.append(key)
                continue

            grp = h5[city][sample]
            if "topology_map" not in grp:
                missing.append(key)
                continue

            topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
            topo_hash = hashlib.sha1(topo.tobytes()).hexdigest()
            altitude_m = extract_scalar(grp, "uav_height")
            groups[(city, topo_hash)].append(
                {
                    "city": city,
                    "sample": sample,
                    "sample_key": key,
                    "altitude_m": altitude_m,
                    "topology_hash": topo_hash,
                }
            )

            if progress_every > 0 and idx % progress_every == 0:
                print(f"[topology-groups] processed {idx}/{len(sample_keys)} samples")

    out_groups: list[dict[str, Any]] = []
    for (city, topo_hash), items in groups.items():
        items.sort(key=lambda item: (item["altitude_m"], item["sample"]))
        rounded_heights = {round(float(item["altitude_m"]), 6) for item in items}
        if len(items) < 2 or len(rounded_heights) < 2:
            continue

        out_groups.append(
            {
                "city": city,
                "topology_hash": topo_hash,
                "items": items,
                "group_size": len(items),
                "min_altitude_m": float(items[0]["altitude_m"]),
                "max_altitude_m": float(items[-1]["altitude_m"]),
                "altitude_span_m": float(items[-1]["altitude_m"] - items[0]["altitude_m"]),
            }
        )

    out_groups.sort(
        key=lambda group: (
            -group["group_size"],
            -group["altitude_span_m"],
            group["city"],
            group["topology_hash"],
        )
    )

    return {
        "groups": out_groups,
        "missing": missing,
    }


def write_report_csv(groups: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "city",
        "topology_hash",
        "group_size",
        "min_altitude_m",
        "max_altitude_m",
        "altitude_span_m",
        "sample",
        "altitude_m",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            base = {
                "city": group["city"],
                "topology_hash": group["topology_hash"],
                "group_size": group["group_size"],
                "min_altitude_m": f"{group['min_altitude_m']:.6f}",
                "max_altitude_m": f"{group['max_altitude_m']:.6f}",
                "altitude_span_m": f"{group['altitude_span_m']:.6f}",
            }
            for item in group["items"]:
                row = dict(base)
                row["sample"] = item["sample"]
                row["altitude_m"] = f"{float(item['altitude_m']):.6f}"
                writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Identify samples that share the same topology_map within a city "
            "but differ in UAV height."
        )
    )
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--hdf5", default="")
    ap.add_argument("--show-limit", type=int, default=20)
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--out", default="", help="Optional CSV report path.")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    config_path = Path(args.config).resolve()
    hdf5_path = (
        Path(args.hdf5).resolve()
        if str(args.hdf5).strip()
        else resolve_hdf5_path(config_path)
    )

    sample_keys = load_sample_keys_from_csv(csv_path)
    report = build_topology_groups(sample_keys, hdf5_path, args.progress_every)
    groups = report["groups"]
    missing = report["missing"]

    print(f"CSV: {csv_path}")
    print(f"HDF5: {hdf5_path}")
    print(f"sample_keys_in_csv={len(sample_keys)}")
    print(f"same-topology-different-height groups={len(groups)}")
    print(f"missing_from_hdf5={len(missing)}")

    for group in groups[: args.show_limit]:
        print(
            f"[group] city={group['city']} size={group['group_size']} "
            f"span={group['altitude_span_m']:.3f}m "
            f"hash={group['topology_hash'][:12]}"
        )
        for item in group["items"]:
            print(
                f"  - {item['sample']} altitude_m={float(item['altitude_m']):.3f}"
            )

    if len(groups) > args.show_limit:
        print(f"... and {len(groups) - args.show_limit} more groups")

    if missing[: args.show_limit]:
        print("samples missing from HDF5:")
        for key in missing[: args.show_limit]:
            print(f"  - {key}")
        if len(missing) > args.show_limit:
            print(f"  ... and {len(missing) - args.show_limit} more")

    if str(args.out).strip():
        out_path = Path(args.out).resolve()
        write_report_csv(groups, out_path)
        print(f"wrote report: {out_path}")


if __name__ == "__main__":
    main()
