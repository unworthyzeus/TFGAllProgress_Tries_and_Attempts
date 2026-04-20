# Try77 Masking Note

Try77 uses the same **masking structure** as Try76, but with a simpler
definition of what counts as a valid target.

## Shared structure

Both tries:

- build `ground_mask` from `topology == 0`
- build a `valid_target` mask
- combine them into `loss_mask`
- apply the training losses only on `loss_mask`

So at the pipeline level, the masking logic is basically the same.

## Try76 rule

Try76 is path-loss specific, so its valid-target filtering is richer:

- target must be finite
- target must be at least the physical floor (`>= 20 dB`)
- it can exclude explicit no-data masks from HDF5
- it can optionally derive invalids from non-ground

That makes Try76 more defensive about sentinel values and dataset-specific
invalid pixels.

## Try77 rule

Try77 is spread specific, so the current rule is simpler:

- target must be finite
- target must be non-negative (`>= 0`)

In code this is:

```python
valid_target = np.isfinite(target) & (target >= 0.0)
loss_mask = ground * valid_target.astype(np.float32)
```

## Practical conclusion

Try77 should be described as using the **same masking pattern** as Try76,
but with a simpler spread-oriented validity rule.
