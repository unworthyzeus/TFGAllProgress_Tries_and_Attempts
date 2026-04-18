# Try74 / Try75 - Registro de cambios de configuración

Fecha: 2026-04-18

Este documento deja constancia de los cambios aplicados en los expertos LoS y NLoS de Try74 y Try75, y de cómo estaban originalmente en `HEAD` antes de los cambios locales.

## Archivos afectados

- TFGSeventyFourthTry74/experiments/seventyfourth_try74_experts/try74_expert_band4555_allcity_los.yaml
- TFGSeventyFourthTry74/experiments/seventyfourth_try74_experts/try74_expert_band4555_allcity_nlos.yaml
- TFGSeventyFifthTry75/experiments/seventyfifth_try75_experts/try75_expert_allcity_los.yaml
- TFGSeventyFifthTry75/experiments/seventyfifth_try75_experts/try75_expert_allcity_nlos.yaml

## Resumen global aplicado

- `model.base_channels` fijado a `60` en los 4 archivos.
- `training.batch_size` mantenido en `1` en todos los casos.
- `training.gradient_accumulation_steps` mantenido en `8`.
- Baseline limpio en los 4 archivos:
  - `pde_residual_loss`: `enabled: false`, `loss_weight: 0.0`
  - `los_highpass_loss`: `enabled: false`, `loss_weight: 0.0`
  - `los_gradient_magnitude_loss`: `enabled: false`, `loss_weight: 0.0`
  - `los_laplacian_pyramid_loss`: `enabled: false`, `loss_weight: 0.0`
  - `nlos_dog_loss`: `enabled: false`, `loss_weight: 0.0`
  - `nlos_gradmag_loss`: `enabled: false`, `loss_weight: 0.0`
  - `nlos_laplacian_pyramid_loss`: `enabled: false`, `loss_weight: 0.0`

## Estado original (HEAD) vs estado actual

### 1) Try74 LoS
Archivo: `TFGSeventyFourthTry74/experiments/seventyfourth_try74_experts/try74_expert_band4555_allcity_los.yaml`

- `base_channels`: 48 -> 60
- `batch_size`: 1 -> 1 (sin cambio)
- `gradient_accumulation_steps`: 8 -> 8 (sin cambio)
- `pde_residual_loss`: deshabilitado en HEAD -> deshabilitado actual (sin cambio)
- Auxiliares LoS/NLoS: habilitadas en HEAD -> deshabilitadas actual

### 2) Try74 NLoS
Archivo: `TFGSeventyFourthTry74/experiments/seventyfourth_try74_experts/try74_expert_band4555_allcity_nlos.yaml`

- `base_channels`: 48 -> 60
- `batch_size`: 1 -> 1 (sin cambio)
- `gradient_accumulation_steps`: 8 -> 8 (sin cambio)
- `pde_residual_loss`: deshabilitado en HEAD -> deshabilitado actual (sin cambio)
- Auxiliares LoS/NLoS: habilitadas en HEAD -> deshabilitadas actual

### 3) Try75 LoS
Archivo: `TFGSeventyFifthTry75/experiments/seventyfifth_try75_experts/try75_expert_allcity_los.yaml`

- `base_channels`: 48 -> 60
- `batch_size`: 1 -> 1 (sin cambio)
- `gradient_accumulation_steps`: 8 -> 8 (sin cambio)
- `pde_residual_loss`: habilitado (0.01) en HEAD -> deshabilitado (0.0) actual
- Auxiliares LoS/NLoS: habilitadas en HEAD -> deshabilitadas actual

### 4) Try75 NLoS
Archivo: `TFGSeventyFifthTry75/experiments/seventyfifth_try75_experts/try75_expert_allcity_nlos.yaml`

- `base_channels`: 48 -> 60
- `batch_size`: 1 -> 1 (sin cambio)
- `gradient_accumulation_steps`: 8 -> 8 (sin cambio)
- `pde_residual_loss`: habilitado (0.01) en HEAD -> deshabilitado (0.0) actual
- Auxiliares LoS/NLoS: habilitadas en HEAD -> deshabilitadas actual

## Nota

La referencia de "original" en este documento se toma de `HEAD` (estado base del repositorio antes de los cambios locales actuales).
