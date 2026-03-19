import json
import glob
from pathlib import Path

BASE = Path(r"c:\TFG\TFGpractice\cluster_outputs")
TRIES = ["TFGThirdTry3", "TFGFourthTry4", "TFGFifthTry5", "TFGSixthTry6"]

def load_best(path: Path):
    best_files = list(path.glob("**/validate_metrics_cgan_best.json"))
    if not best_files:
        return None
    # take most recent
    f = max(best_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(f.read_text())
    pl = data.get("path_loss", {})
    return {
        "file": str(f.relative_to(BASE)),
        "rmse_physical": pl.get("rmse_physical"),
        "unit": pl.get("unit_physical", pl.get("unit")),
        "hybrid_fused": bool(pl.get("hybrid_fused_metrics", False)),
    }

def main():
    rows = []
    for t in TRIES:
        root = BASE / t
        if not root.exists():
            continue
        best = load_best(root)
        rows.append((t, best))
    for t, best in rows:
        print(f"=== {t} ===")
        if not best:
            print("  No best metrics found")
            continue
        print(f"  best file: {best['file']}")
        print(f"  path_loss RMSE: {best['rmse_physical']:.3f} {best['unit']}")
        print(f"  hybrid_fused_metrics: {best['hybrid_fused']}")
        print()

if __name__ == "__main__":
    main()