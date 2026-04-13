# Cluster Stage Chain Safety

## Problem We Hit

In `try62`, `stage1` appeared to stop from Slurm's point of view, so `stage2` was released by `afterany`, but the `stage1` training process kept running on the node and continued writing JSON files.

This caused two bad effects:

1. `stage1` and `stage2` effectively ran at the same time.
2. `stage2` picked an older teacher checkpoint than the later `stage1` JSONs suggested.

## How We Detected It

The timestamps in the exported metrics did not match the Slurm job boundaries:

- Slurm showed `stage1` ended and `stage2` started immediately after.
- But `stage1` kept producing later `validate_metrics_epoch_*.json` files after that point.
- At the same time, `stage2` was already generating its own validation JSONs.

That combination means the original Slurm batch exited, but one or more training children survived as orphan processes on the node.

## Why It Happened

The launcher was calling `torchrun` directly from the batch script, without a robust process cleanup path when the job reached its time limit.

When the batch hit `TIMEOUT`, Slurm considered the batch done, so dependent jobs were allowed to start. But the training children were not always being killed cleanly.

## What We Changed

For the affected launchers, we now:

1. Request a pre-timeout signal with:

   `#SBATCH --signal=B:TERM@90`

2. Run training under `srun` instead of bare `torchrun`.

3. Track the launched child PID.

4. Use a `trap` cleanup handler that:

   - kills the child on `EXIT`, `INT`, or `TERM`
   - waits briefly
   - force-kills if needed
   - deletes the temporary runtime YAML

5. Set:

   `SLURM_STEP_KILL_TIMEOUT=30`

## Operational Rule

Before launching a chained `stage1 -> stage2` experiment on `sert-2001`:

1. Cancel the old chain.
2. Kill possible orphan processes on the node.
3. Wipe the experiment outputs if the previous run may have mixed stages.
4. Relaunch from `stage1`.

If there is any doubt that `stage1` overran past its Slurm end time, do not trust the resulting `stage2` run.

## Practical Symptom Checklist

Suspect overlap immediately if you see any of these:

- `stage2` starts from a teacher that looks older than the latest `stage1` JSON.
- `stage1` JSON timestamps continue after Slurm says the job ended.
- `stage1` and `stage2` validation JSONs alternate in time.
- `sacct` says a job timed out, but the node still behaves as if GPUs are busy.

## Recommendation For Future Chains

- Prefer explicit stage finalization before launching the dependent stage.
- Do not rely on `best_model.pt` alone if the previous stage may still be writing.
- If a run is suspicious, relaunch cleanly instead of trying to salvage mixed outputs.

## Tries Updated With This Protection

- `TFGSixtiethTry60`
- `TFGSixtyFirstTry61`
- `TFGSixtySecondTry62`

