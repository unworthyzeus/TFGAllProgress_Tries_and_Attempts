## Expert Testing Split Rule

When evaluating or refreshing per-expert predictions, always use the official
sample splits defined by Try74 and Try75.

Do not derive evaluation splits from Try58 or Try59 expert datasets, and do not
rebuild train/val/test assignments from spread-expert configs for testing. That
would mix different split definitions and can contaminate evaluation.

Required rule:

- Per-expert testing for path loss, delay spread, or angular spread must be
  mapped onto the preserved Try74/Try75 splits.
- Try58/Try59 experts may provide checkpoints or expert-specific prediction
  logic, but they must not redefine which samples belong to train, val, or
  test for evaluation.
- If there is any conflict, the Try74/Try75 split assignment is the source of
  truth.

Short version: expert testing must follow Try74/Try75 splits to avoid data
contamination.
