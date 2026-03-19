#!/usr/bin/env python3
"""
1) Normalize bare HDF5 paths in YAML to ../Datasets/... **by try folder**:
   - Tries **1–5**: small legacy HDF5 **CKM_Dataset_old_and_small.h5** (same “old/small” set).
     Also repoints mistaken ../CKM_Dataset.h5 or ../CKM_Dataset_180326.h5 in those folders to old_and_small.
   - Tries **6–8**: **CKM_Dataset_180326.h5** (filename with date); fixes mistaken old_and_small → 180326.
   - Tries **9–12**: same as 6–8 (180326); fixes mistaken old_and_small → 180326.

   **CKM_Dataset.h5** = full classic file with no date in the name (not used for tries 1–5 configs here).

2) Add anchor_data_paths_to_config_file to listed tries' config_utils + train scripts if missing.

3) **Prune** — scans **every** run folder under **outputs/** and **cluster_outputs/**:
   Deletes **all** ``epoch_*_cgan.pt`` and ``epoch_N.pt`` (non-GAN); **keeps** ``best_cgan.pt`` / ``best.pt``.
   **Does not delete any .json** (including ``validate_metrics_epoch_*_cgan.json``).

Run from TFGpractice:
  python tools/repoint_hdf5_and_prune_checkpoints.py
  python tools/repoint_hdf5_and_prune_checkpoints.py --dry-run
  python tools/repoint_hdf5_and_prune_checkpoints.py --prune-only      # checkpoints only, no YAML
  python tools/repoint_hdf5_and_prune_checkpoints.py --skip-prune      # YAML + patches only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parent.parent

# Tries 1–5: small legacy HDF5 (not the dated 180326 export).
EARLY_TRY_DIRS = (
    "TFG_FirstTry1",
    "TFGSecondTry2",
    "TFGThirdTry3",
    "TFGFourthTry4",
    "TFGFifthTry5",
)
SIX_TO_EIGHT_DIRS = (
    "TFGSixthTry6",
    "TFGSeventhTry7",
    "TFGEighthTry8",
)
NEW_TRY_DIRS = (
    "TFGNinthTry9",
    "TFGTenthTry10",
    "TFGEleventhTry11",
    "TFGTwelfthTry12",
)

OLD_TRY_NAMES = (
    "TFG_FirstTry1",
    "TFGSecondTry2",
    "TFGThirdTry3",
    "TFGFourthTry4",
    "TFGFifthTry5",
    "TFGSixthTry6",
    "TFGSeventhTry7",
    "TFGEighthTry8",
)

ANCHOR_FUNC = '''def anchor_data_paths_to_config_file(cfg: Dict[str, Any], config_path: str) -> None:
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


'''


def skip_path(p: Path) -> bool:
    parts = set(p.parts)
    return bool(parts & {".venv", "venv", "__pycache__", "node_modules", ".git"})


def _top_try_dir(ypath: Path) -> str:
    try:
        return ypath.relative_to(ROOT).parts[0]
    except ValueError:
        return ""


def migrate_yamls(dry: bool) -> int:
    """Folder-aware: do not blanket-map everything to old_and_small."""
    n = 0
    p_old_small = "hdf5_path: ../Datasets/CKM_Dataset_old_and_small.h5"
    p_classic = "hdf5_path: ../Datasets/CKM_Dataset.h5"
    p_180326 = "hdf5_path: ../Datasets/CKM_Dataset_180326.h5"
    bare_classic = "hdf5_path: CKM_Dataset.h5"
    bare_180326 = "hdf5_path: CKM_Dataset_180326.h5"

    for ypath in ROOT.rglob("*.yaml"):
        if skip_path(ypath):
            continue
        top = _top_try_dir(ypath)
        text = ypath.read_text(encoding="utf-8")
        new = text

        if top in EARLY_TRY_DIRS:
            new = new.replace(bare_classic, p_old_small)
            new = new.replace(bare_180326, p_old_small)
            new = new.replace(p_classic, p_old_small)
            new = new.replace(p_180326, p_old_small)
        elif top in SIX_TO_EIGHT_DIRS:
            new = new.replace(bare_classic, p_180326)
            new = new.replace(bare_180326, p_180326)
            new = new.replace(p_old_small, p_180326)
        elif top in NEW_TRY_DIRS:
            new = new.replace(bare_classic, p_180326)
            if bare_180326 in new:
                new = new.replace(bare_180326, p_180326)
            new = new.replace(p_old_small, p_180326)

        if new != text:
            n += 1
            print(f"YAML: {ypath.relative_to(ROOT)}")
            if not dry:
                ypath.write_text(new, encoding="utf-8", newline="\n")
    return n


def patch_config_utils(path: Path, dry: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    if "anchor_data_paths_to_config_file" in text:
        return False
    if "from typing import Any" not in text:
        text = text.replace(
            "from typing import Dict,",
            "from typing import Any, Dict,",
            1,
        )
    if "from typing import Any" not in text:
        text = text.replace(
            "from typing import Dict\n",
            "from typing import Any, Dict\n",
            1,
        )
    insert_at = text.find("import yaml\n")
    if insert_at < 0:
        return False
    nl = text.find("\n", insert_at) + 1
    new = text[:nl] + "\n" + ANCHOR_FUNC + text[nl:]
    print(f"config_utils: {path.relative_to(ROOT)}")
    if not dry:
        path.write_text(new, encoding="utf-8", newline="\n")
    return True


def patch_train_after_load(text: str, indent: str = "    ") -> str | None:
    if "anchor_data_paths_to_config_file" in text:
        return None
    if "from config_utils import" not in text:
        return None
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    changed_import = False
    changed_call = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("from config_utils import") and "anchor_data_paths_to_config_file" not in line:
            if "load_config" in line:
                line = line.replace(
                    "load_config,",
                    "anchor_data_paths_to_config_file, load_config,",
                    1,
                )
            else:
                line = line.replace(
                    "from config_utils import ",
                    "from config_utils import anchor_data_paths_to_config_file, ",
                    1,
                )
            changed_import = True
        if re.match(r"^(\s*)cfg = load_config\(args\.config\)\s*$", line):
            m = re.match(r"^(\s*)", line)
            ind = m.group(1) if m else indent
            out.append(line)
            i += 1
            out.append(f"{ind}anchor_data_paths_to_config_file(cfg, args.config)\n")
            changed_call = True
            continue
        out.append(line)
        i += 1
    if not changed_import or not changed_call:
        return None
    return "".join(out)


def patch_old_tries(dry: bool) -> None:
    for name in OLD_TRY_NAMES:
        base = ROOT / name
        if not base.is_dir():
            continue
        cu = base / "config_utils.py"
        if cu.is_file():
            patch_config_utils(cu, dry)
        for script in (
            "train_cgan.py",
            "train.py",
            "evaluate_cgan.py",
            "predict_cgan.py",
            "cross_validate_cgan.py",
        ):
            p = base / script
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            if "load_config(args.config)" not in text:
                continue
            new = patch_train_after_load(text)
            if new is None:
                continue
            print(f"train patch: {p.relative_to(ROOT)}")
            if not dry:
                p.write_text(new, encoding="utf-8", newline="\n")


def _under_outputs_or_cluster(p: Path) -> bool:
    parts = p.parts
    return "outputs" in parts or "cluster_outputs" in parts


def prune_dir(d: Path, dry: bool) -> list[Path]:
    """Remove epoch checkpoints only; keep best_cgan.pt / best.pt. Never touches .json."""
    removed: list[Path] = []
    for p in d.glob("epoch_*_cgan.pt"):
        if re.match(r"epoch_\d+_cgan\.pt$", p.name):
            removed.append(p)
    for p in d.glob("epoch_*.pt"):
        if "_cgan" in p.name:
            continue
        if re.match(r"epoch_\d+\.pt$", p.name):
            removed.append(p)
    for p in removed:
        print(f"DELETE pt: {p.relative_to(ROOT)}")
        if not dry:
            p.unlink(missing_ok=True)
    return removed


def prune_all_pts(dry: bool) -> tuple[int, int, int]:
    """Returns (n_pt_removed, n_dirs_scanned, n_pt_files_seen_under_outputs)."""
    dirs_done: Set[Path] = set()
    n = 0
    pt_seen = 0
    for pt in ROOT.rglob("*.pt"):
        if skip_path(pt):
            continue
        if not _under_outputs_or_cluster(pt):
            continue
        pt_seen += 1
        d = pt.parent
        if d in dirs_done:
            continue
        dirs_done.add(d)
        n += len(prune_dir(d, dry))
    return n, len(dirs_done), pt_seen


def patch_slurm_fifthtry5(dry: bool) -> None:
    p = ROOT / "TFGFifthTry5" / "cluster" / "run_fifthtry5_2gpu.slurm"
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    target = "HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_old_and_small.h5}"
    replacements = [
        (
            "HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3/CKM_Dataset.h5}",
            target,
        ),
        (
            "HDF5_PATH=${HDF5_PATH:-/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset.h5}",
            target,
        ),
    ]
    text2 = text
    for a, b in replacements:
        if a in text2:
            text2 = text2.replace(a, b)
    if text2 != text:
        print(f"slurm: {p.relative_to(ROOT)}")
        if not dry:
            p.write_text(text2, encoding="utf-8", newline="\n")


def patch_upload_thirdtry(dry: bool) -> None:
    """Generic uploader: cluster path for tries 1–5 small legacy set."""
    p = ROOT / "cluster" / "upload_and_submit.py"
    if not p.is_file():
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    text2 = text.replace(
        'HDF5_CLUSTER_PATH = "/scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3/CKM_Dataset.h5"',
        'HDF5_CLUSTER_PATH = "/scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_old_and_small.h5"',
    )
    text2 = text2.replace(
        "Dataset on cluster: /scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3/CKM_Dataset.h5 (shared).",
        "Dataset on cluster: /scratch/nas/3/gmoreno/TFGpractice/Datasets/CKM_Dataset_old_and_small.h5 (shared, tries 1-5).",
    )
    if text2 != text:
        print(f"py: {p.relative_to(ROOT)}")
        if not dry:
            p.write_text(text2, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--prune-only",
        action="store_true",
        help="Only prune checkpoints under outputs/ and cluster_outputs/ (no YAML or train patches).",
    )
    ap.add_argument("--skip-prune", action="store_true", help="YAML + patches only, no .pt deletion.")
    args = ap.parse_args()
    dry = args.dry_run
    prune_only = args.prune_only
    if prune_only and args.skip_prune:
        ap.error("--prune-only and --skip-prune are incompatible")

    if not prune_only:
        print("=== YAML ===")
        migrate_yamls(dry)
        print("=== Old tries config_utils / train ===")
        patch_old_tries(dry)
        print("=== Slurm / upload helpers ===")
        patch_slurm_fifthtry5(dry)
        patch_upload_thirdtry(dry)

    if not args.skip_prune:
        print("=== Prune .pt under outputs/ + cluster_outputs/ (all tries; .json never touched) ===")
        n_pt, n_dirs, n_pt_seen = prune_all_pts(dry)
        print(f"Scanned {n_dirs} run folder(s), {n_pt_seen} .pt file(s) seen.")
        print(
            f"Removed {n_pt} epoch checkpoint .pt file(s) (dry={dry}). "
            "Kept: best_cgan.pt / best.pt. All *.json left unchanged."
        )
        if n_pt == 0 and n_pt_seen > 0:
            print("(No epoch_*.pt to remove — only best_*.pt / other .pt in those folders.)")
    else:
        print("=== Prune skipped ===")

    if not prune_only:
        print(
            "Place or symlink datasets under TFGpractice/Datasets/: "
            "CKM_Dataset_old_and_small.h5 (tries 1-5), CKM_Dataset_180326.h5 (tries 6+)."
        )


if __name__ == "__main__":
    main()
