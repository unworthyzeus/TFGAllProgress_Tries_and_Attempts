from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


ROOT = Path(r"c:/TFG/TFGPractice")
DEFAULT_CSV = Path(__file__).with_name("histograms.csv")
DEFAULT_CFG = ROOT / "TFGSeventyFifthTry75" / "experiments" / "seventyfifth_try75_experts" / "try75_expert_allcity_los.yaml"
EXPECTED_KINDS = [
    "target_los",
    "target_nlos",
    "pred_los",
    "pred_nlos",
    "target_delay_spread",
    "pred_delay_spread",
    "target_angular_spread",
    "pred_angular_spread",
]


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


def load_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    bin_cols = [c for c in fieldnames if c.startswith("b") and c[1:].lstrip("-").isdigit()]
    bin_cols.sort(key=lambda c: int(c[1:]))
    return rows, bin_cols


def row_histogram_info(row: dict[str, str], bin_cols: list[str]) -> tuple[int, int, int]:
    total_pixels = int(row.get("total_pixels", "0") or 0)
    counts = np.fromiter((int(row.get(col, "0") or 0) for col in bin_cols), dtype=np.int64, count=len(bin_cols))
    hist_sum = int(counts.sum())
    nonzero_bins = int(np.count_nonzero(counts))
    return total_pixels, hist_sum, nonzero_bins


def sample_key(row: dict[str, str]) -> str:
    return f"{row.get('city', '')}/{row.get('sample', '')}"


def group_rows(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        key = sample_key(row)
        if key not in seen:
            order.append(key)
            seen.add(key)
        grouped[key].append(row)
    return grouped, order


def validate_contiguity(rows: list[dict[str, str]]) -> list[str]:
    positions: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        positions[sample_key(row)].append(idx)
    bad: list[str] = []
    for key, idxs in positions.items():
        start = idxs[0]
        expected = list(range(start, start + len(idxs)))
        if idxs != expected:
            bad.append(key)
    return bad


def first_row_by_kind(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        kind = str(row.get("kind", ""))
        if kind and kind not in out:
            out[kind] = row
    return out


def histogram_counts(row: dict[str, str], bin_cols: list[str]) -> np.ndarray:
    return np.fromiter(
        (int(row.get(col, "0") or 0) for col in bin_cols),
        dtype=np.int64,
        count=len(bin_cols),
    )


def extract_scalar(grp: h5py.Group, name: str, default: float = float("nan")) -> float:
    if name not in grp:
        return default
    return float(np.asarray(grp[name][()]).reshape(()))


def build_topology_height_groups(
    order: list[str],
    hdf5_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_hdf5_samples: list[str] = []
    processed = 0

    with h5py.File(hdf5_path, "r") as h5:
        for key in order:
            city, sample = key.split("/", 1)
            if city not in h5 or sample not in h5[city]:
                missing_hdf5_samples.append(key)
                continue

            grp = h5[city][sample]
            if "topology_map" not in grp:
                missing_hdf5_samples.append(key)
                continue

            topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
            topo_hash = hashlib.sha1(topo.tobytes()).hexdigest()
            groups[(city, topo_hash)].append(
                {
                    "city": city,
                    "sample": sample,
                    "key": key,
                    "altitude_m": extract_scalar(grp, "uav_height"),
                    "topology_hash": topo_hash,
                }
            )

            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"[height-groups] processed {processed}/{len(order)} samples")

    out_groups: list[dict[str, Any]] = []
    for (city, topo_hash), items in groups.items():
        items.sort(key=lambda item: (item["altitude_m"], item["sample"]))
        distinct_heights = {round(float(item["altitude_m"]), 6) for item in items}
        if len(items) < 2 or len(distinct_heights) < 2:
            continue
        out_groups.append(
            {
                "city": city,
                "topology_hash": topo_hash,
                "items": items,
                "group_size": len(items),
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
        "missing_hdf5_samples": missing_hdf5_samples,
        "processed": processed,
    }


def choose_height_pairs(
    items: list[dict[str, Any]],
    pair_mode: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(items) < 2:
        return []
    if pair_mode == "max_span":
        return [(items[0], items[-1])]
    if pair_mode == "adjacent":
        return list(zip(items, items[1:]))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append((items[i], items[j]))
    return pairs


def compare_height_group_histograms(
    grouped: dict[str, list[dict[str, str]]],
    order: list[str],
    bin_cols: list[str],
    hdf5_path: Path,
    progress_every: int,
    kind: str,
    pair_mode: str,
    group_limit: int,
    top_bins: int,
) -> dict[str, Any]:
    topology_report = build_topology_height_groups(order, hdf5_path, progress_every)
    row_lookup = {
        (key, row_kind): row
        for key, rows in grouped.items()
        for row_kind, row in first_row_by_kind(rows).items()
    }

    comparisons: list[dict[str, Any]] = []
    missing_kind_rows: list[str] = []

    for group in topology_report["groups"]:
        for low_item, high_item in choose_height_pairs(group["items"], pair_mode):
            row_a = row_lookup.get((low_item["key"], kind))
            row_b = row_lookup.get((high_item["key"], kind))
            if row_a is None or row_b is None:
                missing_kind_rows.append(
                    f"{group['city']} hash={group['topology_hash'][:12]} "
                    f"{low_item['sample']}->{high_item['sample']} missing kind={kind}"
                )
                continue

            counts_a = histogram_counts(row_a, bin_cols)
            counts_b = histogram_counts(row_b, bin_cols)
            total_a = int(counts_a.sum())
            total_b = int(counts_b.sum())
            delta = counts_b - counts_a
            l1_counts = int(np.abs(delta).sum())

            probs_a = counts_a / max(total_a, 1)
            probs_b = counts_b / max(total_b, 1)
            l1_prob = float(np.abs(probs_b - probs_a).sum())

            peak_a = bin_cols[int(np.argmax(counts_a))] if total_a > 0 else "n/a"
            peak_b = bin_cols[int(np.argmax(counts_b))] if total_b > 0 else "n/a"

            top_changes: list[str] = []
            if top_bins > 0 and delta.size > 0:
                top_idx = np.argsort(np.abs(delta))[-top_bins:][::-1]
                for idx in top_idx:
                    if int(delta[idx]) == 0:
                        continue
                    top_changes.append(f"{bin_cols[int(idx)]}:{int(delta[idx]):+d}")

            comparisons.append(
                {
                    "city": group["city"],
                    "topology_hash": group["topology_hash"],
                    "sample_a": low_item["sample"],
                    "sample_b": high_item["sample"],
                    "altitude_a_m": float(low_item["altitude_m"]),
                    "altitude_b_m": float(high_item["altitude_m"]),
                    "height_delta_m": float(high_item["altitude_m"] - low_item["altitude_m"]),
                    "l1_counts": l1_counts,
                    "l1_prob": l1_prob,
                    "peak_a": peak_a,
                    "peak_b": peak_b,
                    "top_changes": top_changes,
                }
            )

    comparisons.sort(
        key=lambda item: (
            -item["l1_prob"],
            -abs(item["height_delta_m"]),
            item["city"],
            item["sample_a"],
            item["sample_b"],
        )
    )

    lines: list[str] = []
    for item in comparisons[:group_limit]:
        top_changes = ", ".join(item["top_changes"]) if item["top_changes"] else "no bin deltas"
        lines.append(
            f"{item['city']} {item['sample_a']}@{item['altitude_a_m']:.3f}m -> "
            f"{item['sample_b']}@{item['altitude_b_m']:.3f}m "
            f"(dh={item['height_delta_m']:.3f}m, l1_prob={item['l1_prob']:.4f}, "
            f"l1_counts={item['l1_counts']}, peak={item['peak_a']}->{item['peak_b']}, "
            f"deltas={top_changes})"
        )

    return {
        "topology_groups": topology_report["groups"],
        "comparisons": comparisons,
        "lines": lines,
        "missing_hdf5_samples": topology_report["missing_hdf5_samples"],
        "missing_kind_rows": missing_kind_rows,
    }


def check_topology_mask(
    grouped: dict[str, list[dict[str, str]]],
    order: list[str],
    hdf5_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    mismatches: list[str] = []
    missing_hdf5_samples: list[str] = []
    checked = 0

    with h5py.File(hdf5_path, "r") as h5:
        for key in order:
            city, sample = key.split("/", 1)
            if city not in h5 or sample not in h5[city]:
                missing_hdf5_samples.append(key)
                continue
            grp = h5[city][sample]
            if "path_loss" not in grp or "topology_map" not in grp:
                missing_hdf5_samples.append(key)
                continue

            path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
            topo = np.asarray(grp["topology_map"][...], dtype=np.float32)
            if path_loss.shape != topo.shape:
                missing_hdf5_samples.append(key)
                continue

            los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32) > 0.5 if "los_mask" in grp else np.zeros_like(path_loss, dtype=bool)
            valid_ground = np.isfinite(path_loss) & (path_loss > 0.0) & (topo == 0.0)
            expected_target_los = int(np.count_nonzero(valid_ground & los_mask))
            expected_target_nlos = int(np.count_nonzero(valid_ground & (~los_mask)))

            delay_gt = np.asarray(grp["delay_spread"][...], dtype=np.float32) if "delay_spread" in grp else None
            angular_gt = np.asarray(grp["angular_spread"][...], dtype=np.float32) if "angular_spread" in grp else None
            expected_delay = int(np.count_nonzero(valid_ground & np.isfinite(delay_gt) & (delay_gt > 0.0))) if delay_gt is not None and delay_gt.shape == path_loss.shape else None
            expected_angular = int(np.count_nonzero(valid_ground & np.isfinite(angular_gt))) if angular_gt is not None and angular_gt.shape == path_loss.shape else None

            rows_by_kind = first_row_by_kind(grouped.get(key, []))
            comparisons = [
                ("target_los", expected_target_los),
                ("target_nlos", expected_target_nlos),
                ("pred_los", expected_target_los),
                ("pred_nlos", expected_target_nlos),
            ]
            if expected_delay is not None:
                comparisons.append(("target_delay_spread", expected_delay))
            if expected_angular is not None:
                comparisons.append(("target_angular_spread", expected_angular))

            for kind, expected in comparisons:
                row = rows_by_kind.get(kind)
                if row is None:
                    continue
                got = int(row.get("total_pixels", "0") or 0)
                if got != expected:
                    mismatches.append(f"{key} {kind}: csv={got} expected={expected}")

            checked += 1
            if progress_every > 0 and checked % progress_every == 0:
                print(f"[mask-check] checked {checked}/{len(order)} samples")

    return {
        "checked_samples": checked,
        "missing_hdf5_samples": missing_hdf5_samples,
        "mismatches": mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--config", default=str(DEFAULT_CFG), help="Used only to resolve the source HDF5 path.")
    ap.add_argument("--hdf5", default="", help="Optional explicit HDF5 path override.")
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--show-limit", type=int, default=20)
    ap.add_argument(
        "--compare-height-groups",
        action="store_true",
        help="Compare histogram differences for samples sharing the same topology_map but different UAV heights.",
    )
    ap.add_argument(
        "--height-kind",
        default="target_los",
        choices=EXPECTED_KINDS,
        help="Histogram kind to compare inside same-topology different-height groups.",
    )
    ap.add_argument(
        "--height-pair-mode",
        default="max_span",
        choices=["max_span", "adjacent", "all"],
        help="How to pair samples inside each same-topology different-height group.",
    )
    ap.add_argument(
        "--height-group-limit",
        type=int,
        default=10,
        help="Maximum number of same-topology height-difference comparisons to print.",
    )
    ap.add_argument(
        "--height-top-bins",
        type=int,
        default=8,
        help="How many largest per-bin count deltas to show in each height-difference comparison.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    config_path = Path(args.config).resolve()
    hdf5_path = Path(args.hdf5).resolve() if str(args.hdf5).strip() else resolve_hdf5_path(config_path)

    rows, bin_cols = load_csv_rows(csv_path)
    grouped, order = group_rows(rows)
    bad_contiguity = validate_contiguity(rows)

    missing_kind_counter: Counter[str] = Counter()
    complete_8_of_8 = 0
    all_zero_rows: list[str] = []
    constant_rows: list[str] = []
    bad_hist_sum_rows: list[str] = []
    fully_constant_samples: list[str] = []

    for key in order:
        sample_rows = grouped.get(key, [])
        kinds_present = {str(row.get("kind", "")) for row in sample_rows}
        for kind in EXPECTED_KINDS:
            if kind not in kinds_present:
                missing_kind_counter[kind] += 1
        if all(kind in kinds_present for kind in EXPECTED_KINDS) and len(sample_rows) == 8:
            complete_8_of_8 += 1

        nonempty_rows = 0
        constant_nonempty_rows = 0
        for row in sample_rows:
            kind = str(row.get("kind", ""))
            total_pixels, hist_sum, nonzero_bins = row_histogram_info(row, bin_cols)
            if hist_sum != total_pixels:
                bad_hist_sum_rows.append(f"{key} {kind}: total_pixels={total_pixels} hist_sum={hist_sum}")
            if total_pixels > 0:
                nonempty_rows += 1
                if hist_sum == 0:
                    all_zero_rows.append(f"{key} {kind}: total_pixels={total_pixels} but all bins are zero")
                if nonzero_bins == 1:
                    constant_rows.append(f"{key} {kind}: all pixels fell into one bin")
                    constant_nonempty_rows += 1
            elif hist_sum > 0:
                bad_hist_sum_rows.append(f"{key} {kind}: total_pixels=0 but hist_sum={hist_sum}")
        if nonempty_rows > 0 and constant_nonempty_rows == nonempty_rows:
            fully_constant_samples.append(key)

    mask_report = check_topology_mask(grouped, order, hdf5_path, args.progress_every)

    print(f"CSV: {csv_path}")
    print(f"HDF5: {hdf5_path}")
    print(f"rows={len(rows)} samples={len(order)} bins={len(bin_cols)}")
    print(f"samples with 8/8 rows={complete_8_of_8}/{len(order)}")
    print(f"non-contiguous sample groups={len(bad_contiguity)}")
    print(f"rows with histogram sum mismatch={len(bad_hist_sum_rows)}")
    print(f"rows with all-zero histogram despite total_pixels>0={len(all_zero_rows)}")
    print(f"rows with all pixels in one single bin={len(constant_rows)}")
    print(f"samples where every non-empty row is constant={len(fully_constant_samples)}")
    print(f"mask check: checked={mask_report['checked_samples']} missing_hdf5={len(mask_report['missing_hdf5_samples'])} mismatches={len(mask_report['mismatches'])}")

    if missing_kind_counter:
        print("missing kinds by sample count:")
        for kind in EXPECTED_KINDS:
            print(f"  {kind}: {missing_kind_counter.get(kind, 0)}")

    def print_examples(title: str, items: list[str]) -> None:
        if not items:
            return
        print(title)
        for item in items[: args.show_limit]:
            print(f"  - {item}")
        if len(items) > args.show_limit:
            print(f"  ... and {len(items) - args.show_limit} more")

    print_examples("non-contiguous samples:", bad_contiguity)
    print_examples("histogram sum mismatches:", bad_hist_sum_rows)
    print_examples("all-zero suspicious rows:", all_zero_rows)
    print_examples("constant suspicious rows:", constant_rows)
    print_examples("fully constant samples:", fully_constant_samples)
    print_examples("topology mask mismatches:", mask_report["mismatches"])
    print_examples("samples missing in HDF5:", mask_report["missing_hdf5_samples"])

    if args.compare_height_groups:
        height_report = compare_height_group_histograms(
            grouped=grouped,
            order=order,
            bin_cols=bin_cols,
            hdf5_path=hdf5_path,
            progress_every=args.progress_every,
            kind=args.height_kind,
            pair_mode=args.height_pair_mode,
            group_limit=args.height_group_limit,
            top_bins=args.height_top_bins,
        )
        print(
            f"same-topology different-height groups={len(height_report['topology_groups'])} "
            f"comparisons={len(height_report['comparisons'])} "
            f"kind={args.height_kind} pair_mode={args.height_pair_mode}"
        )
        print_examples("height-group histogram comparisons:", height_report["lines"])
        print_examples("height-group missing HDF5 samples:", height_report["missing_hdf5_samples"])
        print_examples("height-group missing kind rows:", height_report["missing_kind_rows"])


if __name__ == "__main__":
    main()
