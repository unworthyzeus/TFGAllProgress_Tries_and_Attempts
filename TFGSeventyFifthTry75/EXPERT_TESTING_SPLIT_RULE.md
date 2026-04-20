## Expert Testing Split Rule

Try75 is one of the split authorities for per-expert testing.

When evaluating expert predictions, preserve the Try74/Try75 sample assignment
and do not replace it with split definitions inherited from Try58 or Try59
expert configs. This is required to avoid evaluation contamination.
