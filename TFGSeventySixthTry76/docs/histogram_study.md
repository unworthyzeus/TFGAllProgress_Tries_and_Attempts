# Try 76 — histogram study (auto-generated)

Source: `tmp_review/histograms_try74/histograms.csv` — aggregated targets and predictions from Try 74 / Try 75 experts.


Bin width = 1 dB (or 1 ns / 1 deg). Summaries use weighted moments over aggregated bin counts. `gmm2` = K=2 Gaussian mixture fit in dB.


## Overall (all cities collapsed)

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 413,884,130 | 6.6 | 9.8 | +2.38 | +6.05 | 0/2/30 | 0 | -19..112 | (pi=0.59, mu=1.3, sigma=1.0) ; (pi=0.41, mu=14.3, sigma=11.6) |
| angular_spread|target_angular_spread | 413,884,130 | 6.6 | 15.7 | +2.87 | +8.93 | 0/0/42 | 0 | 0..180 | (pi=0.84, mu=0.5, sigma=1.0) ; (pi=0.16, mu=38.4, sigma=18.0) |
| delay_spread|pred_delay_spread | 407,801,895 | 8.4 | 9.5 | +2.67 | +7.56 | 2/4/30 | 4 | 0..62 | (pi=0.69, mu=4.1, sigma=1.5) ; (pi=0.31, mu=18.0, sigma=12.3) |
| delay_spread|target_delay_spread | 358,165,824 | 10.4 | 27.7 | +5.47 | +39.76 | 2/4/60 | 2 | 1..359 | (pi=0.86, mu=2.9, sigma=1.2) ; (pi=0.14, mu=57.0, sigma=55.0) |
| path_loss|pred_los | 382,752,235 | 96.4 | 3.8 | -0.89 | +0.94 | 90/98/102 | 98 | 71..104 | (pi=0.35, mu=93.2, sigma=4.0) ; (pi=0.65, mu=98.1, sigma=2.3) |
| path_loss|pred_nlos | 31,131,895 | 36.5 | 22.4 | +0.94 | +0.36 | 8/32/84 | 24 | 0..118 | (pi=0.68, mu=25.4, sigma=11.8) ; (pi=0.32, mu=60.1, sigma=21.2) |
| path_loss|target_los | 382,752,235 | 97.0 | 5.3 | +0.04 | +0.61 | 88/96/106 | 96 | 67..170 | (pi=0.52, mu=96.7, sigma=4.0) ; (pi=0.48, mu=97.2, sigma=6.5) |
| path_loss|target_nlos | 31,131,895 | 107.1 | 4.3 | -0.08 | +2.25 | 100/108/114 | 108 | 74..168 | (pi=0.50, mu=106.5, sigma=5.5) ; (pi=0.50, mu=107.7, sigma=2.4) |

## By topology class (city_type_6) x kind


### dense_block_highrise

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 3,162,975 | 11.6 | 10.1 | +1.02 | +0.56 | 0/10/32 | 0 | -6..68 | (pi=0.67, mu=6.4, sigma=5.1) ; (pi=0.33, mu=21.9, sigma=9.8) |
| angular_spread|target_angular_spread | 3,162,975 | 12.9 | 20.5 | +1.63 | +2.63 | 0/0/50 | 0 | 0..178 | (pi=0.68, mu=0.5, sigma=1.0) ; (pi=0.32, mu=39.1, sigma=17.2) |
| delay_spread|pred_delay_spread | 3,092,143 | 19.6 | 14.6 | +0.81 | -0.62 | 2/12/50 | 10 | 0..62 | (pi=0.57, mu=9.0, sigma=3.8) ; (pi=0.43, mu=33.6, sigma=11.5) |
| delay_spread|target_delay_spread | 2,374,149 | 24.9 | 45.8 | +3.30 | +14.43 | 2/4/120 | 4 | 1..359 | (pi=0.66, mu=3.9, sigma=1.2) ; (pi=0.34, mu=65.2, sigma=60.3) |
| path_loss|pred_los | 2,415,774 | 97.7 | 4.0 | -0.93 | +1.01 | 90/98/102 | 100 | 73..104 | (pi=0.47, mu=95.1, sigma=4.0) ; (pi=0.53, mu=100.0, sigma=2.1) |
| path_loss|pred_nlos | 747,201 | 41.0 | 20.2 | +0.76 | -0.05 | 14/36/82 | 30 | 0..109 | (pi=0.67, mu=30.3, sigma=10.8) ; (pi=0.33, mu=63.2, sigma=16.9) |
| path_loss|target_los | 2,415,774 | 98.2 | 5.9 | +0.23 | +0.73 | 90/98/108 | 98 | 68..153 | (pi=0.50, mu=97.2, sigma=4.0) ; (pi=0.50, mu=99.3, sigma=7.2) |
| path_loss|target_nlos | 747,201 | 107.5 | 4.7 | -0.49 | +2.01 | 100/108/114 | 108 | 82..155 | (pi=0.40, mu=105.8, sigma=6.1) ; (pi=0.60, mu=108.6, sigma=2.9) |

### dense_block_midrise

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 95,440,004 | 9.4 | 12.1 | +1.71 | +2.60 | 0/4/36 | 0 | -14..112 | (pi=0.47, mu=1.2, sigma=1.0) ; (pi=0.53, mu=16.6, sigma=12.8) |
| angular_spread|target_angular_spread | 95,440,004 | 9.2 | 19.0 | +2.29 | +4.95 | 0/0/48 | 0 | 0..178 | (pi=0.79, mu=0.5, sigma=1.0) ; (pi=0.21, mu=42.2, sigma=19.2) |
| delay_spread|pred_delay_spread | 91,044,118 | 6.0 | 6.4 | +2.13 | +4.48 | 2/4/20 | 2 | 0..34 | (pi=0.67, mu=2.8, sigma=1.3) ; (pi=0.33, mu=12.6, sigma=7.5) |
| delay_spread|target_delay_spread | 72,189,190 | 10.1 | 26.1 | +6.40 | +55.92 | 2/4/48 | 4 | 1..359 | (pi=0.83, mu=3.0, sigma=1.2) ; (pi=0.17, mu=43.7, sigma=50.6) |
| path_loss|pred_los | 81,212,002 | 96.4 | 4.1 | -0.89 | +0.78 | 88/98/102 | 98 | 71..104 | (pi=0.39, mu=93.2, sigma=4.1) ; (pi=0.61, mu=98.5, sigma=2.3) |
| path_loss|pred_nlos | 14,228,002 | 36.1 | 22.5 | +0.93 | +0.31 | 8/32/84 | 24 | 0..115 | (pi=0.67, mu=24.8, sigma=11.8) ; (pi=0.33, mu=59.5, sigma=21.3) |
| path_loss|target_los | 81,212,002 | 97.1 | 5.3 | -0.08 | +0.79 | 88/98/106 | 98 | 67..162 | (pi=0.49, mu=96.6, sigma=6.5) ; (pi=0.51, mu=97.5, sigma=3.7) |
| path_loss|target_nlos | 14,228,002 | 106.5 | 4.3 | -0.13 | +2.04 | 100/106/112 | 108 | 74..165 | (pi=0.51, mu=105.8, sigma=5.5) ; (pi=0.49, mu=107.3, sigma=2.5) |

### mixed_compact_lowrise

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 113,816,681 | 6.0 | 10.3 | +2.62 | +7.04 | 0/2/32 | 0 | -19..99 | (pi=0.69, mu=1.3, sigma=1.0) ; (pi=0.31, mu=16.7, sigma=13.3) |
| angular_spread|target_angular_spread | 113,816,681 | 6.1 | 15.2 | +3.01 | +9.86 | 0/0/42 | 0 | 0..179 | (pi=0.86, mu=0.5, sigma=1.0) ; (pi=0.14, mu=40.0, sigma=17.4) |
| delay_spread|pred_delay_spread | 113,427,508 | 4.7 | 3.1 | +2.47 | +7.72 | 2/4/12 | 4 | 0..25 | (pi=0.75, mu=3.5, sigma=1.0) ; (pi=0.25, mu=8.6, sigma=4.0) |
| delay_spread|target_delay_spread | 99,143,237 | 7.5 | 20.8 | +7.25 | +73.85 | 2/2/30 | 2 | 1..359 | (pi=0.89, mu=2.8, sigma=1.2) ; (pi=0.11, mu=44.5, sigma=47.8) |
| path_loss|pred_los | 107,343,579 | 96.4 | 3.8 | -0.90 | +0.94 | 90/98/102 | 98 | 71..104 | (pi=0.35, mu=93.2, sigma=3.9) ; (pi=0.65, mu=98.1, sigma=2.2) |
| path_loss|pred_nlos | 6,473,102 | 32.6 | 21.9 | +1.14 | +0.86 | 6/28/80 | 20 | 0..116 | (pi=0.69, mu=22.0, sigma=10.6) ; (pi=0.31, mu=56.0, sigma=22.0) |
| path_loss|target_los | 107,343,579 | 96.9 | 5.2 | +0.04 | +0.66 | 88/96/106 | 96 | 67..170 | (pi=0.52, mu=96.8, sigma=3.8) ; (pi=0.48, mu=97.1, sigma=6.4) |
| path_loss|target_nlos | 6,473,102 | 107.1 | 3.9 | -0.05 | +3.18 | 100/108/112 | 108 | 79..166 | (pi=0.48, mu=106.4, sigma=5.2) ; (pi=0.52, mu=107.8, sigma=2.0) |

### mixed_compact_midrise

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 70,004,774 | 8.6 | 9.3 | +1.74 | +3.07 | 0/6/30 | 2 | -2..82 | (pi=0.42, mu=1.9, sigma=1.1) ; (pi=0.58, mu=13.5, sigma=9.6) |
| angular_spread|target_angular_spread | 70,004,774 | 9.2 | 17.4 | +2.19 | +5.33 | 0/0/44 | 0 | 0..179 | (pi=0.75, mu=0.5, sigma=1.0) ; (pi=0.25, mu=35.2, sigma=17.4) |
| delay_spread|pred_delay_spread | 68,922,443 | 17.1 | 13.4 | +0.85 | -0.48 | 4/12/44 | 4 | 0..51 | (pi=0.39, mu=5.5, sigma=1.9) ; (pi=0.61, mu=24.5, sigma=12.4) |
| delay_spread|target_delay_spread | 60,603,115 | 19.7 | 41.1 | +3.52 | +16.26 | 2/4/112 | 4 | 1..359 | (pi=0.75, mu=3.2, sigma=1.3) ; (pi=0.25, mu=69.9, sigma=59.1) |
| path_loss|pred_los | 62,770,330 | 96.4 | 3.8 | -0.88 | +0.97 | 90/98/102 | 98 | 71..104 | (pi=0.34, mu=93.2, sigma=4.0) ; (pi=0.66, mu=98.1, sigma=2.4) |
| path_loss|pred_nlos | 7,234,444 | 40.9 | 21.6 | +0.91 | +0.29 | 14/36/88 | 28 | 0..117 | (pi=0.68, mu=30.2, sigma=11.5) ; (pi=0.32, mu=64.1, sigma=20.1) |
| path_loss|target_los | 62,770,330 | 97.0 | 5.7 | +0.15 | +0.60 | 88/96/106 | 96 | 67..167 | (pi=0.50, mu=96.3, sigma=3.9) ; (pi=0.50, mu=97.7, sigma=6.9) |
| path_loss|target_nlos | 7,234,444 | 107.8 | 4.5 | -0.04 | +1.94 | 100/108/116 | 108 | 78..167 | (pi=0.46, mu=107.2, sigma=5.8) ; (pi=0.54, mu=108.3, sigma=2.9) |

### open_sparse_lowrise

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 98,210,129 | 2.9 | 6.0 | +4.37 | +22.50 | 0/0/14 | 0 | -6..95 | (pi=0.83, mu=1.1, sigma=1.0) ; (pi=0.17, mu=12.2, sigma=10.5) |
| angular_spread|target_angular_spread | 98,210,129 | 2.9 | 9.9 | +4.75 | +26.07 | 0/0/28 | 0 | 0..178 | (pi=0.50, mu=2.9, sigma=9.9) ; (pi=0.50, mu=2.9, sigma=9.9) |
| delay_spread|pred_delay_spread | 98,079,703 | 4.9 | 0.7 | -1.19 | +7.78 | 4/4/6 | 4 | 0..17 | (pi=0.50, mu=4.9, sigma=1.0) ; (pi=0.50, mu=4.9, sigma=1.0) |
| delay_spread|target_delay_spread | 92,228,385 | 5.4 | 16.2 | +8.04 | +81.04 | 2/2/8 | 2 | 1..359 | (pi=0.94, mu=2.6, sigma=1.2) ; (pi=0.06, mu=52.9, sigma=48.6) |
| path_loss|pred_los | 96,668,766 | 96.4 | 3.7 | -0.91 | +1.03 | 90/98/102 | 98 | 71..104 | (pi=0.34, mu=93.2, sigma=3.9) ; (pi=0.66, mu=98.0, sigma=2.3) |
| path_loss|pred_nlos | 1,541,363 | 31.9 | 23.0 | +1.17 | +0.82 | 6/26/84 | 18 | 0..116 | (pi=0.68, mu=20.4, sigma=10.6) ; (pi=0.32, mu=56.4, sigma=23.1) |
| path_loss|target_los | 96,668,766 | 96.9 | 5.3 | +0.02 | +0.44 | 88/96/106 | 96 | 67..167 | (pi=0.54, mu=96.0, sigma=4.6) ; (pi=0.46, mu=98.0, sigma=5.8) |
| path_loss|target_nlos | 1,541,363 | 107.5 | 3.8 | +0.08 | +4.23 | 102/108/112 | 108 | 79..168 | (pi=0.40, mu=106.9, sigma=5.4) ; (pi=0.60, mu=108.0, sigma=2.2) |

### open_sparse_vertical

| group | N | mean | std | skew | kurt | p05/p50/p95 | mode | nonzero range | gmm2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| angular_spread|pred_angular_spread | 33,249,567 | 6.4 | 6.2 | +2.22 | +5.97 | 2/4/20 | 2 | -4..75 | (pi=0.69, mu=3.5, sigma=1.9) ; (pi=0.31, mu=12.9, sigma=7.6) |
| angular_spread|target_angular_spread | 33,249,567 | 5.5 | 13.1 | +3.16 | +13.08 | 0/0/36 | 0 | 0..180 | (pi=0.84, mu=0.5, sigma=1.0) ; (pi=0.16, mu=31.2, sigma=16.3) |
| delay_spread|pred_delay_spread | 33,235,980 | 19.0 | 13.4 | +1.78 | +2.08 | 10/12/56 | 10 | 0..60 | (pi=0.59, mu=11.7, sigma=1.5) ; (pi=0.41, mu=29.7, sigma=15.7) |
| delay_spread|target_delay_spread | 31,627,748 | 15.5 | 36.6 | +3.50 | +14.39 | 2/2/104 | 2 | 1..359 | (pi=0.84, mu=2.8, sigma=1.2) ; (pi=0.16, mu=82.4, sigma=55.4) |
| path_loss|pred_los | 32,341,784 | 96.1 | 3.6 | -0.93 | +0.96 | 90/96/100 | 98 | 74..104 | (pi=0.36, mu=93.1, sigma=3.8) ; (pi=0.64, mu=97.8, sigma=2.1) |
| path_loss|pred_nlos | 907,783 | 40.5 | 23.3 | +0.88 | +0.13 | 10/36/90 | 30 | 0..118 | (pi=0.68, mu=28.5, sigma=12.0) ; (pi=0.32, mu=66.1, sigma=20.8) |
| path_loss|target_los | 32,341,784 | 96.6 | 5.4 | +0.10 | +0.37 | 88/96/106 | 96 | 70..159 | (pi=0.51, mu=95.9, sigma=4.1) ; (pi=0.49, mu=97.4, sigma=6.4) |
| path_loss|target_nlos | 907,783 | 108.6 | 4.3 | +0.25 | +2.02 | 102/108/116 | 108 | 80..154 | (pi=0.43, mu=108.5, sigma=2.2) ; (pi=0.57, mu=108.8, sigma=5.4) |

## Per-city quick stats (path_loss targets only)

| city | target_los mean/std/p50 | target_nlos mean/std/p50 |
|---|---|---|
| Bilbao | 97.4/5.7/98 | 108.1/4.8/108 |
| Chiang Mai | 96.6/5.6/96 | 108.3/4.9/108 |
| Cleveland | 96.6/5.5/96 | 107.3/4.0/108 |
| Dhaka | 97.0/5.6/96 | 107.9/4.7/108 |
| Dubrovnik | 96.6/5.4/96 | 106.8/4.4/108 |
| Fortaleza | 97.3/4.7/98 | 106.1/3.8/106 |
| Marrakesh | 97.1/5.6/96 | 107.8/4.2/108 |
| Mexico City | 97.0/5.6/96 | 106.9/4.4/108 |
| Munich | 97.2/5.7/96 | 107.5/4.3/108 |
| Nazareth | 96.6/5.5/96 | 107.5/4.5/108 |
| Nice | 96.9/5.6/96 | 107.0/4.1/108 |
| Salvador | 96.9/5.7/96 | 106.7/4.3/108 |
| Segovia | 97.0/5.4/96 | 107.4/4.8/108 |
| Tunis | 96.8/4.7/98 | 106.2/3.8/106 |
| Victoria | 97.1/5.5/96 | 106.9/3.7/108 |