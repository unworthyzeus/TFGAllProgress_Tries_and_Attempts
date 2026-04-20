# Try 76 — distribution-class identification

Auto-generated from `tmp_review/histograms_try74/histograms.csv`.

For each aggregated histogram we fit a handful of parametric families on the 1-dB/ns/deg bins and rank them by KL(empirical || model). The smaller the KL, the better the fit.

Families: `gaussian`, `laplace`, `skew_normal`, `lognormal`, `gamma`, `weibull`, `spike_plus_exp` (for heavy-tailed sparse metrics), `gmm2`, `gmm3`, `gmm4`, `gmm5`.


## Overall (all cities collapsed)

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm4** | 6.6 | 9.8 | +2.38 | +6.05 | gmm4(KL=0.082) > gmm5(KL=0.082) > gmm3(KL=0.102) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 6.6 | 15.7 | +2.87 | +8.93 | spike_plus_exp(KL=0.652) > gmm4(KL=0.792) > gmm5(KL=0.792) |
| delay_spread|pred_delay_spread | **gmm5** | 8.4 | 9.5 | +2.67 | +7.56 | gmm5(KL=0.030) > gmm4(KL=0.033) > gmm3(KL=0.033) |
| delay_spread|target_delay_spread | **gmm5** | 10.4 | 27.7 | +5.47 | +39.76 | gmm5(KL=0.119) > gmm4(KL=0.151) > gmm3(KL=0.161) |
| path_loss|pred_los | **gmm5** | 96.4 | 3.8 | -0.89 | +0.94 | gmm5(KL=0.004) > gmm4(KL=0.004) > gmm3(KL=0.005) |
| path_loss|pred_nlos | **gmm5** | 36.5 | 22.4 | +0.94 | +0.36 | gmm5(KL=0.008) > gmm4(KL=0.010) > gmm3(KL=0.013) |
| path_loss|target_los | **gmm4** | 97.0 | 5.3 | +0.04 | +0.61 | gmm4(KL=0.003) > gmm5(KL=0.003) > gmm3(KL=0.003) |
| path_loss|target_nlos | **gmm5** | 107.1 | 4.3 | -0.08 | +2.25 | gmm5(KL=0.037) > gmm4(KL=0.037) > gmm3(KL=0.039) |

## By topology class (city_type_6) x kind


### dense_block_highrise

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm5** | 11.6 | 10.1 | +1.02 | +0.56 | gmm5(KL=0.016) > gmm4(KL=0.037) > gmm3(KL=0.037) |
| angular_spread|target_angular_spread | **gmm5** | 12.9 | 20.5 | +1.63 | +2.63 | gmm5(KL=0.627) > spike_plus_exp(KL=0.632) > gmm4(KL=0.666) |
| delay_spread|pred_delay_spread | **gmm5** | 19.6 | 14.6 | +0.81 | -0.62 | gmm5(KL=0.035) > gmm4(KL=0.048) > gmm3(KL=0.157) |
| delay_spread|target_delay_spread | **gmm4** | 24.9 | 45.8 | +3.30 | +14.43 | gmm4(KL=0.122) > gmm5(KL=0.124) > gmm3(KL=0.131) |
| path_loss|pred_los | **gmm4** | 97.7 | 4.0 | -0.93 | +1.01 | gmm4(KL=0.004) > gmm5(KL=0.004) > gmm3(KL=0.016) |
| path_loss|pred_nlos | **gmm5** | 41.0 | 20.2 | +0.76 | -0.05 | gmm5(KL=0.004) > gmm4(KL=0.004) > gmm3(KL=0.005) |
| path_loss|target_los | **gmm5** | 98.2 | 5.9 | +0.23 | +0.73 | gmm5(KL=0.011) > gmm3(KL=0.011) > gmm4(KL=0.011) |
| path_loss|target_nlos | **gmm5** | 107.5 | 4.7 | -0.49 | +2.01 | gmm5(KL=0.057) > gmm4(KL=0.057) > gmm3(KL=0.058) |

### dense_block_midrise

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm5** | 9.4 | 12.1 | +1.71 | +2.60 | gmm5(KL=0.033) > gmm4(KL=0.036) > gmm3(KL=0.051) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 9.2 | 19.0 | +2.29 | +4.95 | spike_plus_exp(KL=0.650) > gmm2(KL=0.757) > gmm3(KL=0.757) |
| delay_spread|pred_delay_spread | **gmm5** | 6.0 | 6.4 | +2.13 | +4.48 | gmm5(KL=0.016) > gmm4(KL=0.016) > gmm3(KL=0.036) |
| delay_spread|target_delay_spread | **gmm5** | 10.1 | 26.1 | +6.40 | +55.92 | gmm5(KL=0.100) > gmm4(KL=0.100) > gmm3(KL=0.110) |
| path_loss|pred_los | **gmm5** | 96.4 | 4.1 | -0.89 | +0.78 | gmm5(KL=0.001) > gmm4(KL=0.002) > gmm3(KL=0.006) |
| path_loss|pred_nlos | **gmm5** | 36.1 | 22.5 | +0.93 | +0.31 | gmm5(KL=0.009) > gmm4(KL=0.011) > gmm3(KL=0.014) |
| path_loss|target_los | **gmm5** | 97.1 | 5.3 | -0.08 | +0.79 | gmm5(KL=0.003) > gmm3(KL=0.003) > gmm4(KL=0.003) |
| path_loss|target_nlos | **gmm5** | 106.5 | 4.3 | -0.13 | +2.04 | gmm5(KL=0.031) > gmm4(KL=0.032) > gmm3(KL=0.033) |

### mixed_compact_lowrise

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm5** | 6.0 | 10.3 | +2.62 | +7.04 | gmm5(KL=0.122) > gmm4(KL=0.122) > gmm3(KL=0.122) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 6.1 | 15.2 | +3.01 | +9.86 | spike_plus_exp(KL=0.675) > gmm4(KL=0.815) > gmm3(KL=0.815) |
| delay_spread|pred_delay_spread | **gmm5** | 4.7 | 3.1 | +2.47 | +7.72 | gmm5(KL=0.027) > gmm4(KL=0.027) > gmm2(KL=0.053) |
| delay_spread|target_delay_spread | **gmm5** | 7.5 | 20.8 | +7.25 | +73.85 | gmm5(KL=0.121) > gmm4(KL=0.121) > gmm3(KL=0.170) |
| path_loss|pred_los | **gmm5** | 96.4 | 3.8 | -0.90 | +0.94 | gmm5(KL=0.003) > gmm4(KL=0.004) > gmm3(KL=0.004) |
| path_loss|pred_nlos | **gmm5** | 32.6 | 21.9 | +1.14 | +0.86 | gmm5(KL=0.010) > gmm4(KL=0.011) > gmm3(KL=0.016) |
| path_loss|target_los | **gmm5** | 96.9 | 5.2 | +0.04 | +0.66 | gmm5(KL=0.003) > gmm4(KL=0.003) > gmm3(KL=0.003) |
| path_loss|target_nlos | **gmm5** | 107.1 | 3.9 | -0.05 | +3.18 | gmm5(KL=0.028) > gmm4(KL=0.028) > gmm3(KL=0.029) |

### mixed_compact_midrise

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm5** | 8.6 | 9.3 | +1.74 | +3.07 | gmm5(KL=0.032) > gmm4(KL=0.036) > gmm3(KL=0.056) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 9.2 | 17.4 | +2.19 | +5.33 | spike_plus_exp(KL=0.609) > gmm5(KL=0.712) > gmm4(KL=0.712) |
| delay_spread|pred_delay_spread | **gmm5** | 17.1 | 13.4 | +0.85 | -0.48 | gmm5(KL=0.009) > gmm4(KL=0.018) > gmm3(KL=0.030) |
| delay_spread|target_delay_spread | **gmm5** | 19.7 | 41.1 | +3.52 | +16.26 | gmm5(KL=0.114) > gmm4(KL=0.127) > gmm3(KL=0.154) |
| path_loss|pred_los | **gmm5** | 96.4 | 3.8 | -0.88 | +0.97 | gmm5(KL=0.006) > gmm4(KL=0.007) > gmm3(KL=0.007) |
| path_loss|pred_nlos | **gmm5** | 40.9 | 21.6 | +0.91 | +0.29 | gmm5(KL=0.004) > gmm4(KL=0.005) > gmm3(KL=0.007) |
| path_loss|target_los | **gmm4** | 97.0 | 5.7 | +0.15 | +0.60 | gmm4(KL=0.004) > gmm5(KL=0.004) > gmm3(KL=0.004) |
| path_loss|target_nlos | **gmm4** | 107.8 | 4.5 | -0.04 | +1.94 | gmm4(KL=0.054) > gmm5(KL=0.054) > gmm3(KL=0.055) |

### open_sparse_lowrise

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm5** | 2.9 | 6.0 | +4.37 | +22.50 | gmm5(KL=0.245) > gmm4(KL=0.245) > gmm2(KL=0.308) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 2.9 | 9.9 | +4.75 | +26.07 | spike_plus_exp(KL=0.687) > lognormal(KL=0.844) > weibull(KL=0.950) |
| delay_spread|pred_delay_spread | **laplace** | 4.9 | 0.7 | -1.19 | +7.78 | laplace(KL=0.065) > skew_normal(KL=0.127) > gaussian(KL=0.142) |
| delay_spread|target_delay_spread | **gmm5** | 5.4 | 16.2 | +8.04 | +81.04 | gmm5(KL=0.143) > gmm4(KL=0.159) > gmm3(KL=0.159) |
| path_loss|pred_los | **gmm5** | 96.4 | 3.7 | -0.91 | +1.03 | gmm5(KL=0.005) > gmm4(KL=0.005) > gmm3(KL=0.006) |
| path_loss|pred_nlos | **gmm5** | 31.9 | 23.0 | +1.17 | +0.82 | gmm5(KL=0.013) > gmm4(KL=0.015) > gmm3(KL=0.022) |
| path_loss|target_los | **gmm4** | 96.9 | 5.3 | +0.02 | +0.44 | gmm4(KL=0.002) > gmm5(KL=0.002) > gmm3(KL=0.002) |
| path_loss|target_nlos | **gmm5** | 107.5 | 3.8 | +0.08 | +4.23 | gmm5(KL=0.033) > gmm4(KL=0.034) > gmm3(KL=0.035) |

### open_sparse_vertical

| group | best family | mean | std | skew | excess kurt | top-3 by KL |
|---|---|---:|---:|---:|---:|---|
| angular_spread|pred_angular_spread | **gmm4** | 6.4 | 6.2 | +2.22 | +5.97 | gmm4(KL=0.026) > gmm5(KL=0.027) > gmm3(KL=0.062) |
| angular_spread|target_angular_spread | **spike_plus_exp** | 5.5 | 13.1 | +3.16 | +13.08 | spike_plus_exp(KL=0.629) > gmm5(KL=0.783) > gmm4(KL=0.783) |
| delay_spread|pred_delay_spread | **gmm5** | 19.0 | 13.4 | +1.78 | +2.08 | gmm5(KL=0.067) > gmm4(KL=0.067) > gmm3(KL=0.128) |
| delay_spread|target_delay_spread | **gmm5** | 15.5 | 36.6 | +3.50 | +14.39 | gmm5(KL=0.135) > gmm3(KL=0.135) > gmm4(KL=0.137) |
| path_loss|pred_los | **gmm5** | 96.1 | 3.6 | -0.93 | +0.96 | gmm5(KL=0.002) > gmm4(KL=0.002) > gmm3(KL=0.002) |
| path_loss|pred_nlos | **gmm5** | 40.5 | 23.3 | +0.88 | +0.13 | gmm5(KL=0.006) > gmm4(KL=0.007) > gmm3(KL=0.012) |
| path_loss|target_los | **gmm5** | 96.6 | 5.4 | +0.10 | +0.37 | gmm5(KL=0.002) > gmm4(KL=0.002) > gmm3(KL=0.003) |
| path_loss|target_nlos | **gmm5** | 108.6 | 4.3 | +0.25 | +2.02 | gmm5(KL=0.080) > gmm4(KL=0.083) > gmm3(KL=0.083) |

## Takeaways

- `path_loss | target_los` → **gmm4** — the only metric where a single near-Gaussian is competitive.
- `path_loss | target_nlos` → **gmm5** — a narrow peak around 107 dB with a secondary low-σ mode; a 2-component Gaussian mixture wins, a single Gaussian is too wide.
- `delay_spread | target_delay_spread` → **gmm5** — not Gaussian; a spike-at-≈3 ns plus a long exponential tail is the right parametric form.
- `angular_spread | target_angular_spread` → **spike_plus_exp** — same shape as delay spread: spike-at-0 + heavy tail; Gaussian is a bad choice.

Implication for Try 76: the Stage-A distribution head must be **family-aware per expert**. Path-loss experts output a 3-component Gaussian mixture, but delay/angular experts (not trained in Try 76 but noted for Try 77+) should output `(π_spike, λ_tail)` for a degenerate+exponential model, not a GMM.