## Expert Testing Split Rule

Per-expert testing must use the official Try74/Try75 split assignment.

Do not use Try59 dataset-derived train/val/test partitions as the evaluation
authority when validating expert outputs. Try59 checkpoints can still be used
for prediction, but the tested sample membership must come from Try74/Try75 so
that evaluation stays contamination-free.
