#!/usr/bin/env python3
"""One-shot: copy TFGSixtySixthTry66 → TFGSixtyEighthTry68, rename try66→try68, apply PMHHNet HF stem fix.

Run from repo root:  python TFGpractice/scripts/bootstrap_try68_from_try66.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".slurm",
    ".md",
    ".tex",
    ".json",
    ".toml",
    ".txt",
    ".sh",
    ".cfg",
}

# Order: longer / more specific first.
CONTENT_REPLACEMENTS: list[tuple[str, str]] = [
    ("TFGSixtySixthTry66", "TFGSixtyEighthTry68"),
    ("sixtysixth_try66", "sixtyeighth_try68"),
    ("try66_expert", "try68_expert"),
    ("try66_topology", "try68_topology"),
    ("logs_train_try66", "logs_train_try68"),
    ("logs_cleanup_try66", "logs_cleanup_try68"),
    ("logs_train_try66_chain", "logs_train_try68_chain"),
    ("t66-cleanup", "t68-cleanup"),
    ("t66-chain", "t68-chain"),
    ("ckm-t66-cleanup", "ckm-t68-cleanup"),
    ("ckm-t66-4gpu", "ckm-t68-4gpu"),
    ("ckm-t66-2gpu", "ckm-t68-2gpu"),
    ("Try66 chain", "Try68 chain"),
    ("Try 66", "Try 68"),
    ("try66 chain", "try68 chain"),
    ("ckm_t66_", "ckm_t68_"),
    ("submit_try66", "submit_try68"),
    ("try66_", "try68_"),
    ("t66-", "t68-"),
    ("MASTER_PORT:-29966", "MASTER_PORT:-29988"),
    ("MASTER_PORT=${MASTER_PORT:-29966}", "MASTER_PORT=${MASTER_PORT:-29988}"),
]

PMHHNET_OLD = """    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._resolve_scalar_cond(x, scalar_cond)

        x0 = self._run(self.stem, x)
        x0 = self.film_stem(x0, cond)
        e1 = self.se1(self.film_e1(self._run(self.stage1, x0), cond))
        e2 = self.se2(self.film_e2(self._run(self.stage2, e1), cond))
        e3 = self.se3(self.film_e3(self._run(self.stage3, e2), cond))
        e4 = self.se4(self.film_e4(self._run(self.stage4, e3), cond))
        c4 = self._run(self.context, e4)
        c4 = self.film_context(c4, cond)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(e3) + self._upsample_like(p4, e3))
        p2 = self.smooth2(self.lat2(e2) + self._upsample_like(self.top3_to_2(p3), e2))
        p1 = self.smooth1(self.lat1(e1) + self._upsample_like(self.top2_to_1(p2), e1))

        hf = self.hf_project(self._high_frequency_map(x))
        hf = self.film_hf(hf, cond)"""

PMHHNET_NEW = """    def forward(self, x: torch.Tensor, scalar_cond: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._resolve_scalar_cond(x, scalar_cond)

        # PMHNet path: stem(x) + hf_project(Laplacian|x|); FiLM must keep this sum (Try 66 PMHHNet dropped it).
        hf = self._run(self.hf_project, self._high_frequency_map(x))
        x0 = self._run(self.stem, x) + hf
        x0 = self.film_stem(x0, cond)
        e1 = self.se1(self.film_e1(self._run(self.stage1, x0), cond))
        e2 = self.se2(self.film_e2(self._run(self.stage2, e1), cond))
        e3 = self.se3(self.film_e3(self._run(self.stage3, e2), cond))
        e4 = self.se4(self.film_e4(self._run(self.stage4, e3), cond))
        c4 = self._run(self.context, e4)
        c4 = self.film_context(c4, cond)

        p4 = self.lat4(c4)
        p3 = self.smooth3(self.lat3(e3) + self._upsample_like(p4, e3))
        p2 = self.smooth2(self.lat2(e2) + self._upsample_like(self.top3_to_2(p3), e2))
        p1 = self.smooth1(self.lat1(e1) + self._upsample_like(self.top2_to_1(p2), e1))

        hf = self.film_hf(hf, cond)"""


def _port_bump(text: str, old_base: int, delta: int) -> str:
    for i in range(6):
        text = text.replace(str(old_base + i), str(old_base + delta + i))
    return text


def _transform_file(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    out = raw
    for a, b in CONTENT_REPLACEMENTS:
        out = out.replace(a, b)
    out = out.replace("default=30266", "default=30286")
    out = _port_bump(out, 30066, 100)  # chain expert ports → 30166..30171
    if path.name == "model_pmhhnet.py" and PMHHNET_OLD in out:
        out = out.replace(PMHHNET_OLD, PMHHNET_NEW)
    if out != raw:
        path.write_text(out, encoding="utf-8", newline="\n")


def _rename_path(path: Path, root: Path) -> Path:
    name = path.name
    new_name = name
    for old, new in (
        ("sixtysixth_try66_experts", "sixtyeighth_try68_experts"),
        ("sixtysixth_try66_classifier", "sixtyeighth_try68_classifier"),
        ("try66_expert_registry", "try68_expert_registry"),
        ("try66_topology_classifier", "try68_topology_classifier"),
        ("try66_expert_", "try68_expert_"),
        ("run_sixtysixth_try66_", "run_sixtyeighth_try68_"),
        ("submit_try66_", "submit_try68_"),
        ("generate_try66_configs", "generate_try68_configs"),
        ("plot_try66_metrics", "plot_try68_metrics"),
        ("_fetch_try66_logs", "_fetch_try68_logs"),
        ("_relaunch_try66", "_relaunch_try68"),
    ):
        if old in new_name:
            new_name = new_name.replace(old, new)
    if new_name != name:
        dest = path.with_name(new_name)
        path.rename(dest)
        return dest
    return path


def main() -> int:
    practice = Path(__file__).resolve().parents[1]
    src = practice / "TFGSixtySixthTry66"
    dst = practice / "TFGSixtyEighthTry68"
    if not src.is_dir():
        print(f"Missing source: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)
    print(f"Copying {src.name} -> {dst.name}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )

    # Rename directories bottom-up
    all_dirs = sorted([p for p in dst.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True)
    for d in all_dirs:
        new_name = d.name
        for old, new in (
            ("sixtysixth_try66_experts", "sixtyeighth_try68_experts"),
            ("sixtysixth_try66_classifier", "sixtyeighth_try68_classifier"),
        ):
            if old in new_name:
                new_name = new_name.replace(old, new)
        if new_name != d.name:
            d.rename(d.with_name(new_name))

    # Rename files
    for f in list(dst.rglob("*")):
        if f.is_file():
            _rename_path(f, dst)

    # Transform text files
    for f in dst.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() in TEXT_SUFFIXES or f.name.endswith(".slurm"):
            _transform_file(f)

    print("Done. HF stem fix applied in model_pmhhnet.py if the old forward block matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
