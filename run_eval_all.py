"""Run validation for all downloaded try57/try68/try73 checkpoints sequentially."""
from __future__ import annotations
import datetime, io, json, os, subprocess, sys, threading, time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(BASE, "cluster_outputs")
EVAL_SCRIPT = os.path.join(BASE, "tmp_eval_los.py")

JOBS = [
    # (try_dir, config_rel, expert_tag, ckpt_rel)
    # ("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_dense_block_highrise.yaml",   "try57_dense_block_highrise",    "TFGFiftySeventhTry57/fiftyseventhtry57_expert_dense_block_highrise/best_model.pt"),
    #("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_dense_block_midrise.yaml",     "try57_dense_block_midrise",     "TFGFiftySeventhTry57/fiftyseventhtry57_expert_dense_block_midrise/best_model.pt"),
    #("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_mixed_compact_lowrise.yaml",   "try57_mixed_compact_lowrise",   "TFGFiftySeventhTry57/fiftyseventhtry57_expert_mixed_compact_lowrise/best_model.pt"),
    #("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_mixed_compact_midrise.yaml",   "try57_mixed_compact_midrise",   "TFGFiftySeventhTry57/fiftyseventhtry57_expert_mixed_compact_midrise/best_model.pt"),
    #("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_open_sparse_lowrise.yaml",     "try57_open_sparse_lowrise",     "TFGFiftySeventhTry57/fiftyseventhtry57_expert_open_sparse_lowrise/best_model.pt"),
    # ("TFGFiftySeventhTry57", "experiments/fiftyseventhtry57_partitioned_stage1/fiftyseventhtry57_expert_open_sparse_vertical.yaml",    "try57_open_sparse_vertical",    "TFGFiftySeventhTry57/fiftyseventhtry57_expert_open_sparse_vertical/best_model.pt"),

    # ("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_dense_block_highrise.yaml",   "try68_dense_block_highrise",   "TFGSixtyEighthTry68/try68_expert_dense_block_highrise/best_model.pt"),
    #("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_dense_block_midrise.yaml",    "try68_dense_block_midrise",    "TFGSixtyEighthTry68/try68_expert_dense_block_midrise/best_model.pt"),
    #("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_mixed_compact_lowrise.yaml",  "try68_mixed_compact_lowrise",  "TFGSixtyEighthTry68/try68_expert_mixed_compact_lowrise/best_model.pt"),
    #("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_mixed_compact_midrise.yaml",  "try68_mixed_compact_midrise",  "TFGSixtyEighthTry68/try68_expert_mixed_compact_midrise/best_model.pt"),
    #("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_open_sparse_lowrise.yaml",    "try68_open_sparse_lowrise",    "TFGSixtyEighthTry68/try68_expert_open_sparse_lowrise/best_model.pt"),
    # ("TFGSixtyEighthTry68",  "experiments/sixtyeighth_try68_experts/try68_expert_open_sparse_vertical.yaml",   "try68_open_sparse_vertical",   "TFGSixtyEighthTry68/try68_expert_open_sparse_vertical/best_model.pt"),

    # ("TFGSeventyThirdTry73",  "experiments/seventythird_try73_experts/try73_expert_open_sparse_lowrise.yaml",            "try73_open_sparse_lowrise",         "TFGSeventyThirdTry73/try73_expert_open_sparse_lowrise/best_model.pt"),
    # ("TFGSeventyThirdTry73",  "experiments/seventythird_try73_experts/try73_expert_open_sparse_vertical.yaml",           "try73_open_sparse_vertical",        "TFGSeventyThirdTry73/try73_expert_open_sparse_vertical/best_model.pt"),

    # ("TFGSeventyFourthTry74", "experiments/seventyfourth_try74_experts/try74_expert_band4555_allcity_los.yaml", "try74_band4555_allcity_los",        "TFGSeventyFourthTry74/try74_expert_band4555_allcity_los/best_model.pt"),

    ("TFGSeventyFifthTry75",  "experiments/seventyfifth_try75_experts/try75_expert_allcity_los.yaml",           "try75_allcity_los",                 "TFGSeventyFifthTry75/try75_expert_allcity_los_small/best_model.pt"),
]

results_dir = os.path.join(BASE, "tmp_eval_results")
os.makedirs(results_dir, exist_ok=True)


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def run_streaming(cmd, cwd):
    """Run subprocess, stream stderr live (progress bars etc.), capture stdout for JSON."""
    stdout_buf = io.StringIO()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    def drain_stderr():
        for line in proc.stderr:
            print(f"  > {line}", end="", flush=True)

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()
    for line in proc.stdout:
        stdout_buf.write(line)
    proc.wait()
    t.join()
    return proc.returncode, stdout_buf.getvalue()


active = [(td, cr, tag, ck) for td, cr, tag, ck in JOBS if not ck.startswith("#")]
n_total = sum(1 for _, _, _, ck in active if os.path.exists(os.path.join(OUT_BASE, ck)))
n_done = 0

for try_dir, config_rel, tag, ckpt_rel in active:
    ckpt_abs = os.path.join(OUT_BASE, ckpt_rel)
    if not os.path.exists(ckpt_abs):
        print(f"[{ts()}] SKIP {tag} — no checkpoint")
        continue

    out_path = os.path.join(results_dir, f"{tag}.json")
    if os.path.exists(out_path):
        n_done += 1
        print(f"[{ts()}] SKIP {tag} — already evaluated ({n_done}/{n_total})")
        continue

    cwd = os.path.join(BASE, try_dir)
    cmd = [sys.executable, EVAL_SCRIPT, "--config", config_rel, "--checkpoint", ckpt_abs, "--split", "val", "--device", "cuda"]
    n_done += 1
    print(f"\n[{ts()}] -- [{n_done}/{n_total}] {tag} ------------------", flush=True)
    t0 = time.time()

    rc, stdout = run_streaming(cmd, cwd)
    elapsed = time.time() - t0

    if rc != 0:
        print(f"[{ts()}] ERROR {tag} failed in {elapsed:.0f}s (rc={rc})")
        with open(out_path + ".err", "w") as f:
            f.write(stdout)
        continue

    with open(out_path, "w") as f:
        f.write(stdout)
    print(f"[{ts()}] OK  {tag}  {elapsed:.0f}s", flush=True)


# ── Summary ────────────────────────────────────────────────────────────────
print("\n============ SUMMARY ============")

def walk(d, pre=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += walk(v, f"{pre}.{k}" if pre else k)
    elif isinstance(d, (int, float)):
        out.append((pre, d))
    return out

for _, _, tag, _ in active:
    out_path = os.path.join(results_dir, f"{tag}.json")
    if not os.path.exists(out_path):
        print(f"  {tag:<42s}  MISSING")
        continue
    try:
        d = json.load(open(out_path))
        kv = dict(walk(d))
        rmse_all = kv.get("path_loss.rmse_physical") or kv.get("focus.path_loss.rmse_physical")
        rmse_los = (kv.get("focus.regimes.path_loss__los__LoS.rmse_physical")
                    or kv.get("focus.regimes.path_loss__prior__los__LoS.rmse_physical"))
        rmse_nlos = (kv.get("focus.regimes.path_loss__los__NLoS.rmse_physical")
                     or kv.get("focus.regimes.path_loss__prior__los__NLoS.rmse_physical"))
        parts = [f"rmse_all={rmse_all:.3f}" if rmse_all else "rmse_all=?"]
        if rmse_los:  parts.append(f"los={rmse_los:.3f}")
        if rmse_nlos: parts.append(f"nlos={rmse_nlos:.3f}")
        print(f"  {tag:<42s}  {'  '.join(parts)}")
    except Exception as e:
        print(f"  {tag}: parse error {e}")
