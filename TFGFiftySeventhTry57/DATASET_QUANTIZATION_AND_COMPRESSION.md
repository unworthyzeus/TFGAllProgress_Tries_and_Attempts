# Dataset Quantization And Compression

This note records the current CKM dataset storage format so future cache and
dataset work can stay aligned with what already compresses well in practice.

## Dataset

- file:
  - `CKM_Dataset_270326.h5`
- measured size:
  - `2422199445 bytes`
  - `2.256 GiB`
- total content:
  - `16180` samples
  - `66` cities

## Per-dataset storage pattern

- `path_loss`
  - `uint8`
  - `gzip` level `4`
  - chunk `(513, 513)`
- `los_mask`
  - `uint8`
  - `gzip` level `4`
  - chunk `(513, 513)`
- `angular_spread`
  - `uint8`
  - `gzip` level `4`
- `delay_spread`
  - `uint16`
  - `gzip` level `4`
- `topology_map`
  - `float32`
  - `gzip` level `4`

## Why this matters

The HDF5 stays small because it combines:

- quantized storage where possible
- structured map data that compresses well
- chunked datasets
- `gzip` compression per dataset

That is why the real dataset can sit around `2.256 GiB` even though the raw
tensor volume would be much larger.

## Important takeaway

- `delay_spread` being `uint16` is important and worth preserving as a design
  clue for future compact caches or auxiliary targets.
- Shared prior caches should mimic the HDF5 philosophy:
  - quantized storage when possible
  - chunked datasets
  - `gzip` compression
  - one consolidated HDF5 instead of thousands of loose `.pt` files
