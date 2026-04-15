# Analisis del Dataset y Estrategia de Entrenamiento - Try 66

Analisis completo del `CKM_Dataset_270326.h5` con conclusiones contrastadas
con la literatura, benchmarks de estado del arte, y recomendaciones
concretas para el entrenamiento.

---

## 1. Estructura del Dataset

| Campo | Shape | dtype | Descripcion |
|---|---|---|---|
| `topology_map` | 513x513 | float32 | Alturas de edificios en metros (0 = suelo) |
| `path_loss` | 513x513 | **uint8** | Path loss en dB, resolucion de 1 dB |
| `los_mask` | 513x513 | uint8 | 0/1 binario: LoS o NLoS |
| `uav_height` | 1x1 | float32 | Altura de antena UAV en metros |
| `angular_spread` | 513x513 | uint8 | Spread angular (no usado) |
| `delay_spread` | 513x513 | uint16 | Delay spread (no usado) |

**Totales:**
- **66 ciudades**, **16,180 muestras**
- Resolucion: 513x513 = 263,169 pixeles/muestra
- ~4,260 millones de pixeles totales

**Ciudades mas grandes:** Abu Dhabi (1,105), Rio de Janeiro (985), Shanghai
(770), Santiago (700), Mexico City (650)

**Ciudades mas pequenas:** Thane (20), Bratislava (25), Chefchaouen (25),
Segovia (30), Toledo (30)

---

## 2. Distribucion de Alturas UAV

```
Rango:  11.6 - 477.5 m
Media:  114.0 m
Mediana: 75.7 m
```

| Bin | Muestras | % |
|---|---|---|
| 0-30 m | 942 | 5.8% |
| 30-50 m | 3,030 | 18.7% |
| 50-75 m | 4,030 | 24.9% |
| 75-100 m | 2,528 | 15.6% |
| 100-150 m | 1,898 | 11.7% |
| 150-200 m | 819 | 5.1% |
| 200-300 m | 1,671 | 10.3% |
| 300-500 m | 1,262 | 7.8% |

### Conclusion: distribucion MUY sesgada y de cola pesada

La mediana es 75.7 m pero hay un 18.1% de muestras por encima de 200 m.
El modelo tiene que funcionar bien en todo el rango 15-475 m, pero la
mayoria de los datos estan en 30-100 m.

**Implicacion:** Las alturas altas (>200 m) son regimenes faciles - casi
todo es LoS, path loss ~ free-space. Las alturas bajas (<50 m) son las
dificiles - muchos edificios estan a la misma altura o por encima de la
antena. El sinusoidal embedding es critico para que el modelo distinga
estos regimenes.

**Fuente:** Borhani et al. (arXiv:2511.10763, 2025) confirman que el PLE
NLoS decrece de ~5 a baja altitud a ~2.5 a alta altitud. El desafio
esta en las alturas bajas.

---

## 3. Path Loss - La Verdad del Ground Truth

### 3.1 Cuantizacion: uint8 = 1 dB de resolucion

El path loss se almacena como `uint8`, lo que significa:
- **Rango efectivo: 65-184 dB** (valores realmente usados)
- **Resolucion: 1 dB** por paso de cuantizacion
- **Solo 120 valores unicos** de 256 posibles

Esto tiene consecuencias directas:

1. **El error irreducible teorico es ~0.29 dB RMS** por redondeo
   (sqrt(1/12) ~ 0.29 para cuantizacion uniforme).
2. **No tiene sentido perseguir RMSE < 1 dB** - el ground truth no tiene
   esa precision.
3. **Label smoothing NO ayuda aqui** - los targets ya son discretos con
   saltos de 1 dB; suavizarlos solo anade ruido artificial.

**Fuente:** La cuantizacion uniforme introduce error RMS = D/sqrt(12)
(teoria de senales clasica; Widrow & Kollar, "Quantization Noise," 2008).

### 3.2 Distribucion de valores (solo pixeles de suelo)

| Rango dB | Pixeles | % del total con datos |
|---|---|---|
| 60-70 | 3,080 | 0.0% |
| 70-80 | 3,966,672 | 0.2% |
| 80-90 | 200,402,785 | 8.4% |
| **90-100** | **1,419,225,489** | **59.4%** |
| **100-110** | **686,961,650** | **28.8%** |
| 110-120 | 76,543,438 | 3.2% |
| 120-130 | 1,267,249 | 0.1% |
| 130+ | 157,579 | 0.0% |

### Conclusion: el 88.2% de todos los datos con GT estan entre 90-110 dB

El path loss es una distribucion **extremadamente concentrada** en 90-110
dB. El modelo necesita muy poca capacidad para aprender esta distribucion
central - el reto esta en los extremos (60-90 y 110-150).

**Implicacion para el training:**
- MSE/Huber loss en este rango producen gradientes dominados por la masa
  central (90-110 dB). Las colas (NLoS profundo, LoS cercano) contribuyen
  poco al gradiente total.
- Esto explica por que el NLoS RMSE no baja: los pixeles NLoS con datos
  reales (100-130 dB) son solo el ~3% del total.

---

## 4. Pixeles NLoS con path_loss=0: Edificios por diseno, NO datos faltantes

### 4.1 Correccion importante respecto a analisis anteriores

Los pixeles NLoS con `path_loss=0` **NO son datos faltantes del ray-tracer**.
Son pixeles dentro de la mascara de edificios (`topology_map > 0`), donde
**por diseno** no hay receptor - un edificio no recibe senal en su interior.

El modelo **no tiene por que predecir path loss en estos pixeles**.
Ya se excluyen correctamente mediante la building mask tanto en
training loss como en validation metrics.

### 4.2 Breakdown LoS vs NLoS (solo pixeles de suelo validos)

| Regimen | Pixeles ground | % total ground |
|---|---|---|
| LoS | 2,186,427,815 | 67.6% |
| NLoS | 1,045,859,875 | 32.4% |

De los NLoS de suelo:
- ~202M tienen `path_loss > 0` (datos validos del ray-tracer)
- ~844M tienen `path_loss = 0` (dentro de edificios, excluidos por mask)

### 4.3 Implicacion real para NLoS RMSE

Los NLoS RMSE de 25-38 dB reportados en validacion son **sobre pixeles
NLoS de suelo con datos validos** (no edificios). Esto significa que el
modelo genuinamente falla en esos pixeles por la dificultad intrinseca
del NLoS, no por confusion con edificios.

La relacion LoS:NLoS validos es aproximadamente **10:1** en muestras
`open_sparse`, lo que sigue implicando un desbalance significativo de
gradiente.

### 4.4 Distribucion NLoS con datos

| Rango dB | Pixeles NLoS | % del NLoS con datos |
|---|---|---|
| 90-100 | 11,472,834 | 5.7% |
| **100-110** | **139,191,195** | **68.9%** |
| **110-120** | **50,348,220** | **24.9%** |
| 120-130 | 794,993 | 0.4% |

El NLoS que si tiene datos esta en **100-120 dB**, lo cual es solo 10-20
dB por encima del LoS (90-100 dB). Estos son NLoS "suaves" - difraccion
de un solo edificio, reflexiones cercanas.

---

## 5. Benchmarks de Estado del Arte

### 5.1 ICASSP 2023 First Pathloss Radio Map Prediction Challenge (Outdoor)

**Dataset:** RadioMap3DSeer - 256x256 px, Tx en rooftop, alturas
variables de edificios (6.6-19.8 m). Normalizacion 0-1 con rango ~36 dB.
Los pixeles de edificio se fuerzan a 0 en prediccion y GT (error=0 por diseno).

| Metodo | RMSE (norm.) | RMSE (dB aprox.) | Notas |
|---|---|---|---|
| **REM-Net+** | 0.0349 | **~1.26 dB** | Late submission, April 2024 |
| **PMNet** (H/8 x W/8) | 0.0383 | **~1.38 dB** | 1st place ICASSP 2023 |
| Agile (KL, LoS) | 0.0451 | ~1.62 dB | |
| PPNet | 0.0507 | ~1.83 dB | |

**Fuente:** radiomapchallenge.github.io/results.html; Yapar et al.,
arXiv:2310.07658; Lee et al., ICASSP 2023.

**Conversion nota:** La normalizacion usa f = max{(PL - PL_trnc)/(M1 - PL_trnc), 0}
con M1 = -75 dB, PL_trnc = -111 dB -> rango = 36 dB. RMSE_dB ~ RMSE_norm * 36.
Ademas, los edificios (~30% de pixeles) contribuyen 0 error, inflando la
metrica favorablemente. El RMSE real sobre pixeles no-edificio es ~15-20% mayor.

### 5.2 RadioMapSeer 3D con height encoding

**Dataset:** Mismo RadioMap3DSeer, entrenamiento con info de altura.

| Metodo | RMSE (dB) | Notas |
|---|---|---|
| RadioUNet-3D (12 height slices) | **0.87 dB** | Height-encoded input |
| RadioUNet-naive (sin height) | 1.26 dB | Solo BW city map |

**Fuente:** Yapar et al., arXiv:2212.11777, Sec. IV, Fig. 3.

### 5.3 ICASSP 2025 First Indoor Pathloss Challenge

**Dataset:** Indoor, multifrecuencia (868 MHz, 1.8 GHz, 3.5 GHz),
antenas directivas. MUCHO mas dificil que outdoor rooftop.

| Metodo | RMSE (dB) | Notas |
|---|---|---|
| Wenlihan_Lu (1st) | **9.41 dB** | |
| IPP-Net (2nd) | 9.50 dB | |
| TerRaIn (3rd) | 10.33 dB | |
| TransPathNet (4th) | 10.40 dB | Two-stage coarse-to-fine |

**Fuente:** indoorradiomapchallenge.github.io/results.html;
arxiv:2501.16023 (TransPathNet).

### 5.4 Weighting Map ResNet (Gao et al., 2026)

Mejora sobre PPNet, RPNet, ViT de **1.2-3.0 dB** usando Tx/Rx depth maps
+ weighting map + distance map. 60% menos FLOPs.

**Fuente:** Gao et al., arXiv:2601.08436, Jan 2026.

### 5.5 Resumen nuestros resultados (CKM_Dataset_270326.h5)

| Try | Expert | Overall RMSE | LoS RMSE | NLoS RMSE | Notas |
|---|---|---|---|---|---|
| 42 | global | 19.17 dB | 3.84 dB | 33.40 dB | PMNet + calibrated prior |
| 22 | global | 19.94 dB | 3.78 dB | 34.43 dB | U-Net bilinear baseline |
| 60 | open_sparse_vertical | 18.41 dB | 3.92 dB | 49.47 dB | Partitioned expert |
| **64** | **open_sparse_lowrise** | **7.76 dB** | **2.97 dB** | **24.91 dB** | **Best single expert** |
| 64 | open_sparse_vertical | 13.86 dB | 3.69 dB | 36.22 dB | |
| 65 | open_sparse_lowrise | 11.36 dB | 4.62 dB | 37.88 dB | Grokking stress test |

### 5.6 Analisis de viabilidad del objetivo de 5 dB RMSE global

**El objetivo de 5 dB RMSE global es ambicioso pero no imposible.** Contexto:

1. **Nuestro mejor expert (Try 64 open_sparse_lowrise) ya tiene 7.76 dB.**
   Esto es un solo expert en el regimen mas facil. El desafio es replicar
   esto en los expertos densos y verticales.

2. **El ICASSP 2023 challenge consigue ~1.4 dB** pero en un dataset muy
   diferente: 256x256, alturas fijas de edificios, Tx fijo en rooftop,
   un solo tipo de entorno. Nuestro dataset es 513x513, alturas UAV de
   12-478 m, 66 ciudades diversas.

3. **El indoor challenge (ICASSP 2025) tiene ~9.4 dB** en un problema con
   multifrecuencia y antenas directivas. Nuestro problema es mas facil
   que indoor (outdoor, frecuencia fija, antena isotropica) pero mas
   dificil que el ICASSP 2023 outdoor (mas ciudades, alturas variables).

4. **La clave es NLoS.** Con LoS ya en ~3 dB y NLoS en ~25-37 dB, el
   overall RMSE esta dominado por NLoS. Para bajar a 5 dB global:
   - LoS: mantener en ~3 dB (ya conseguido)
   - NLoS: bajar a ~10-12 dB (requiere mejora de 2-3x)
   - Esto requiere que NLoS pase de ~25 dB a ~12 dB

5. **Hoja de ruta realista:**

| Horizonte | Overall RMSE | LoS | NLoS | Que se necesita |
|---|---|---|---|---|
| Try 66 (actual) | 7-9 dB | 2.5-3 dB | 18-25 dB | FiLM + CutMix + NLoS reweight |
| Optimizado | 5-7 dB | 2-3 dB | 12-18 dB | + ensemble expertos + TTA |
| Limite teorico | ~3-4 dB | ~2 dB | ~8-10 dB | + mejor prior NLoS + datos NLoS |

**Conclusion: 5 dB global es posible como meta intermedia** para los
expertos faciles (lowrise, sparse). Para un 5 dB **global ponderado
por todos los expertos**, se requeriria avances significativos en NLoS,
posiblemente con mejor NLoS prior o datos sinteticos adicionales.

---

## 6. Analisis de Edificios

- **Altura media de edificios:** 15.2 m (p10=5.7, p50=12.9, p90=27.8)
- **Densidad de edificios:** variable, desde ~10% hasta >60%

### Implicacion para tx_depth_map

Con `building_height - antenna_height`:
- A 30 m de altura: la mayoria de edificios (p50=12.9 m) estan **por
  debajo** -> tx_depth negativo -> mayormente LoS
- A 15 m de altura: muchos edificios a la misma altura o por encima ->
  tx_depth cerca de 0 o positivo -> NLoS frecuente
- A 200+ m: todos los edificios muy por debajo -> tx_depth muy negativo ->
  casi todo LoS

Esto confirma que el tx_depth_map es una senal rica para el modelo.

---

## 7. Recomendaciones Basadas en los Datos

### 7.1 Sobre el overfitting (12 canales en tries 55-60)

**Causa raiz identificada:** Con 16,180 muestras y un modelo con FiLM en
cada capa + SE attention + base_channels=40/48, el ratio
parametros/muestras es alto. Y solo ~20% de los pixeles NLoS de suelo
tienen datos validos, asi que el modelo efectivo ve poca diversidad NLoS.

**Soluciones IMPLEMENTADAS en Try 66:**

| Tecnica | Config | Fuente |
|---|---|---|
| **Dropout 2D** (0.12) | `dropout: 0.12` | Subido de 0.08 |
| **Weight decay** (0.015) | `weight_decay: 0.015` | Subido de 0.01; AdamW desacopla |
| **CutMix** (p=0.25) | `cutmix_prob: 0.25` | Yun et al. (arXiv:1905.04899); MICCAI 2024 |
| **Geometric augmentation** | `hflip=0.5, vflip=0.5, rot90=0.5` | Re-habilitada (ver seccion 7.6) |
| **EMA decay** (0.995) | `ema_decay: 0.995` | arXiv:2411.18704 |
| **Gradient accumulation** (4) | `gradient_accumulation_steps: 4` | Simula batch=4 con batch_size=1 |

**Descartadas:**
- **Stochastic depth / DropPath:** Requiere cambios arquitecturales profundos
  en PMHHNet (bloques residuales con drop probability creciente). El beneficio
  marginal no justifica la complejidad con las otras regularizaciones activas.
- **Reducir base_channels:** Mantenemos 40/48 porque la capacidad es necesaria
  para el FiLM conditioning (8 injection points). Si hay overfitting, CutMix +
  dropout + weight decay son preferibles a reducir capacidad.

**CutMix es la recomendacion mas fuerte:** En MICCAI 2024, Zhang et al.
("Cut to the Mix", Paper #674) demostraron que CutMix supera a todas las
augmentaciones elaboradas en datasets limitados de segmentacion.

**Implementacion CutMix con batch_size=1:** Se mantiene un buffer del sample
anterior. Con probabilidad `cutmix_prob`, se genera una caja aleatoria
(distribucion Beta(1,1) para el ratio lambda), y se reemplazan input, target
y mask en la caja con el sample anterior. Esto es equivalente al CutMix
original pero funciona con batch_size=1.

### 7.2 Sobre la convergencia NLoS

**El problema real:** La relacion LoS:NLoS validos es ~10:1. El gradiente
esta completamente dominado por LoS. Los pixeles NLoS con datos reales
(100-120 dB) son solo ~3% del total de pixeles con GT.

**Solucion IMPLEMENTADA en Try 66:**

1. **NLoS pixel reweighting (factor 2.5x):** Los pixeles NLoS (LoS channel
   <= 0.5) reciben 2.5x peso en la mascara de loss. Esto se aplica
   multiplicando `mask = mask * (1 + (factor - 1) * nlos_pixels)`.
   El efecto: un pixel NLoS contribuye 2.5x mas al gradiente que un LoS.

2. **Huber loss (delta=6.0 dB):** Los errores NLoS grandes (>6 dB) se
   linearizan, evitando que dominen con gradientes cuadraticos. Los errores
   LoS pequenos (<6 dB) mantienen gradiente cuadratico para precision fina.

**Descartadas por ahora:**

- **Focal loss adaptado a regresion:** `w(e) = |e|^gamma`. En teoria
  prometedor, pero combinar focal + Huber + NLoS reweight es excesiva
  complejidad. Si NLoS no mejora con la config actual, se puede probar.
  **Fuente:** Lin et al., arXiv:1708.02002; Ribeiro et al., arXiv:2408.14718.

- **Separate NLoS head:** Head auxiliar para "excess path loss" NLoS.
  Requiere cambios arquitecturales. Deferido a Try 67+ si NLoS sigue >15 dB.

- **Aceptar el ceiling parcial:** Con solo ~202M pixeles NLoS validos
  y 1 dB de resolucion, hay un limite. Pero el NLoS de 25 dB actual
  tiene mucho margen de mejora antes de ese limite.

### 7.3 Sobre la cuantizacion uint8

**Recomendacion: NO cambiar nada en la loss por esto.**

Con 1 dB de resolucion y rango 65-184 dB:
- El error irreducible es 0.29 dB - insignificante vs los 8-40 dB RMSE actuales
- Ordinal regression (BEL; Cheng et al., ICLR 2022) podria teoricamente
  ayudar pero anade complejidad innecesaria dado que MSE/Huber ya funcionan
  bien para esta resolucion

**El verdadero impacto del uint8:** La prediccion del modelo es float32,
pero el target es entero. Cuando el modelo predice 97.3 dB y el GT es 97
dB, el error de 0.3 dB es puro ruido de cuantizacion. No perseguir RMSE
< 1 dB.

### 7.4 Sobre el tail refiner (Stage 2) — ELIMINADO en Try 66

**Decision: Stage 2 eliminado. Entrenamos directo a 513x513.**

**Por que se elimino:**

1. En Try 49-64, el refiner solo mejoraba **0.2-0.5 dB** consistentemente.
2. Con uint8 GT (1 dB resolucion), la correccion del refiner esta dentro
   del ruido de cuantizacion — no se puede distinguir mejora real de ruido.
3. El refiner aprende un "residual del residual": la correccion sobre
   una prediccion ya razonable. Estos residuos son de ~1-5 dB.
4. El pipeline de dos etapas duplica el tiempo de entrenamiento y anade
   riesgo de propagacion de errores (Stage 1 malo → Stage 2 malo).
5. Entrenar directo a 513x513 con batch_size=1 + gradient accumulation
   es viable y conserva detalles de alta frecuencia desde el inicio.

**UNetResidualRefinerH** (definido en model_pmhhnet.py) sigue disponible
en el codigo si se quiere re-activar en futuros tries, pero no se usa
en la configuracion de Try 66.

### 7.5 Sobre el prior

**Por que no ayuda tanto como esperado:**

El prior (two-ray, COST231) produce una estimacion razonable para LoS
(~90-100 dB con FSPL) pero una estimacion **muy mala** para NLoS porque
no tiene informacion de los obstaculos especificos. El prior predice un
NLoS "promedio" por city_type, pero la realidad depende de la geometria
exacta de los edificios frente al Tx.

**El prior es mas util cuanto mas simple sea la geometria:**
- En `open_sparse_lowrise`: prior bueno (pocos obstaculos)
- En `dense_block_highrise`: prior pobre (geometria compleja)

### 7.6 Sobre la augmentacion geometrica — VERIFICADA SEGURA

La augmentacion (hflip, vflip, rot90) fue deshabilitada en tries anteriores
por sospecha de que el prior channel y la mask no se flipean de la misma
manera, confundiendo al modelo. **Esto se ha auditado y es incorrecto.**

**La augmentacion ES consistente.** Verificado en `data_utils.py`:

1. `_apply_sync_aug()` (linea ~2184) recibe una **lista unica** de TODOS
   los tensores espaciales y aplica la MISMA transformacion aleatoria a
   todos:
   - `input_tensor` (topology map normalizado)
   - `los_input_tensor` (LoS mask)
   - `distance_map_tensor` (distancia al centro)
   - `formula_input_tensor` (prior calculado)
   - `prior_confidence_tensor`
   - obstruction features
   - `raw_path_loss_tensor`, `path_loss_invalid_mask`, `non_ground_mask`
   - target tensors (path_loss GT)

2. **El prior se computa ANTES de la augmentacion** y luego se augmenta
   junto con todo lo demas. El prior usa `los_formula_tensor` (= LoS mask
   pre-augmentacion) para calcular path loss con la geometria correcta,
   y luego ambos (prior y LoS) reciben el mismo flip/rotation.

3. **Canales post-augmentacion** (tx_depth_map, elevation_angle_map):
   - `tx_depth_map`: usa `input_tensor` (YA augmentado) * scale, asi que
     es espacialmente consistente con los datos augmentados.
   - `elevation_angle_map`: usa `_compute_distance_map_2d()` que genera
     un mapa radialmente simetrico desde el centro (Tx). Todas las
     transformaciones D4 (flips + rot90/180/270) dejan un mapa radial
     **invariante**, asi que no necesita ser augmentado.

**Checklist para futuras modificaciones del pipeline:**

Si se anade un nuevo canal espacial, verificar:
- [ ] Si es **pre-augmentacion**: anadirlo al stack de `_apply_sync_aug`
- [ ] Si es **post-augmentacion**: debe calcularse desde tensores YA
  augmentados, o ser radialmente simetrico (invariante a D4)
- [ ] Si usa **datos cacheados** (e.g. prior cache): el cache debe ser
  del dato NO augmentado, y la augmentacion se aplica despues de cargar

**NUNCA** computar un canal espacial desde datos pre-augmentacion y
concatenarlo a canales post-augmentacion sin transformarlo. Eso rompe
la consistencia espacial y el modelo aprende correlaciones espurias.

---

## 8. Configuracion Optima Recomendada para Try 66

### Basada en todo lo anterior:

```yaml
# ==============================================================
# CONFIGURACION FINAL IMPLEMENTADA (generate_try66_configs.py)
# ==============================================================

# Resolucion: 513x513 DIRECTA, sin Stage 2 refiner
image_size: 513
batch_size: 1
gradient_accumulation_steps: 4  # effective batch = 4

# Anti-overfitting
dropout: 0.12
weight_decay: 0.015
cutmix_prob: 0.25   # CutMix en 25% de batches (mezcla con sample anterior)
cutmix_alpha: 1.0   # distribución Beta(1,1) = uniforme

# Anti-NLoS convergence
loss_type: huber
huber_delta: 6.0           # lineariza errores >6 dB (NLoS)
nlos_reweight_factor: 2.5  # NLoS pixels reciben 2.5x peso en la loss mask
# Nota: la reweighting modifica la mascara de loss, NO los datos de entrada

# Learning rate - Cosine Annealing Warm Restarts (SGDR)
lr: 3e-4
lr_scheduler: cosine_warm_restarts
cosine_T0: 40
cosine_Tmult: 2
lr_eta_min: 1e-6

# EMA
ema_decay: 0.995

# Modelo
base_channels: 40-48    # 40 para sparse experts, 48 para dense/compact
hf_channels: 16-20      # proporcional
use_se_attention: true
se_reduction: 4

# FiLM conditioning (CRITICO: sinusoidal embedding)
use_scalar_film: true
sinusoidal_embed_dim: 64   # 32 frequency bands
# 8 FiLM injection points: stem, e1, e2, e3, e4, context, hf, fused

# Experts: 6 topology classes
experts:
  - open_sparse_lowrise   (base=40, hf=16)
  - open_sparse_vertical  (base=40, hf=16)
  - mixed_compact_lowrise (base=40, hf=16)
  - mixed_compact_midrise (base=48, hf=20)
  - dense_block_midrise   (base=48, hf=20)
  - dense_block_highrise  (base=48, hf=20)

# Epochs y early stopping
epochs: 300
early_stopping_patience: 50
rewind_to_best_model: true
```

### Timing estimado

Con 513x513, batch_size=1, grad_accum=4:
- ~1,000-2,500 muestras por expert (con partition filter)
- ~1,000-2,500 steps/epoch, ~250-625 optimizer steps
- ~5-12 min/epoch en 4 GPUs (mayor por imagen grande, pero aceptable)

### Justificacion: por que NO Stage 2

1. En Try 49-64, el tail refiner mejoraba solo 0.2-0.5 dB
2. Con uint8 GT (1 dB resolucion), la correccion de Stage 2 esta por debajo del ruido de cuantizacion
3. Single-stage simplifica el pipeline y evita error de propagacion
4. Entrenar directo a 513x513 conserva detalles de alta frecuencia desde el inicio

### Validacion JSON: esquema limpio

```yaml
metrics:
  path_loss:         {rmse_physical, mae_physical, count, unit}
  path_loss_513:     {rmse_physical, mae_physical, count}  # full-res
  train_path_loss:   {rmse_physical, mae_physical}  # online weights, train()
  prior_path_loss:   {rmse_physical, mae_physical}
  improvement_vs_prior: {rmse_gain_db, mae_gain_db}
focus:
  topology_class: str
  regimes:
    path_loss__los__LoS:  {rmse_physical, mae_physical, fraction_of_valid_pixels}
    path_loss__los__NLoS: {rmse_physical, mae_physical, fraction_of_valid_pixels}
    path_loss__city_type__*: ...
    path_loss__antenna_bin__*: ...
support:
  sample_count, los_fraction, nlos_fraction
runtime:
  generator_loss, learning_rate, train_seconds, val_seconds
  loss_components: {final_loss, multiscale_loss, nlos_focus_loss, ...}
checkpoint:
  epoch, best_epoch, best_score
selection:
  metric, current_score, is_best_epoch
selection_proxy:
  overall_rmse_physical, nlos_rmse_physical, alpha, composite_nlos_weighted_rmse
model_info:
  val_uses_ema: bool
  ema_decay: float
  train_metrics_use_online_weights: true
  note: "train RMSE usa online weights en train(); val RMSE usa EMA en eval()"
```

**Cambios vs Try 64:**
- **Eliminado** `path_loss_128` (era duplicado de `path_loss` para 128x128; ahora es 513 directo)
- **Eliminado** `focus.routed_city_type` (campo muerto)
- **Nuevo** `model_info` con flag EMA — explica por que val < train
- **Nuevo** plotter: `scripts/plot_try66_metrics.py` con 5 paneles

### Por que val RMSE < train RMSE (no es bug)

1. **EMA vs online**: val usa EMA model (promedio exponencial, mas suave), train usa pesos online sin promediar
2. **eval() vs train()**: dropout esta activo en train, off en eval — train RMSE tiene ruido extra
3. Esto es esperado y deseable — significa que EMA funciona

---

## 9. Fuentes Completas

### Analisis de datos
- Dataset: `CKM_Dataset_270326.h5`, 66 ciudades, 16,180 muestras
- Analisis propio con `_inspect_fast.py` (scan completo del HDF5)

### Benchmarks y challenges

1. **ICASSP 2023 Challenge:** radiomapchallenge.github.io/results.html;
   Yapar et al., "The First Pathloss Radio Map Prediction Challenge,"
   arXiv:2310.07658, Oct 2023.

2. **RadioMap3DSeer Dataset:** Yapar et al., "Dataset of Pathloss and
   ToA Radio Maps With Localization Application," arXiv:2212.11777, 2022.
   Normalizacion: f = max{(PL-PL_trnc)/(M1-PL_trnc), 0}, rango 3D = 36 dB.

3. **PMNet:** Lee et al., "PMNet: Large-Scale Channel Prediction System
   for ICASSP 2023 First Pathloss Radio Map Prediction Challenge,"
   ICASSP 2023.

4. **REM-Net+:** "REM-Net+: 3D Radio Environment Map Construction Guided
   by Radio Propagation Model," TechRxiv, April 2024.

5. **ICASSP 2025 Indoor Challenge:**
   indoorradiomapchallenge.github.io/results.html

6. **TransPathNet:** Li et al., "TransPathNet: A Novel Two-Stage Framework
   for Indoor Radio Map Prediction," arXiv:2501.16023, Jan 2025.

7. **IPP-Net:** "IPP-Net: A Generalizable Deep Neural Network Model for
   Indoor Pathloss Radio Map Prediction," ICASSP 2025.

8. **Weighting Map:** Gao et al., "Effective outdoor pathloss prediction:
   A multi-layer segmentation approach with weighting map,"
   arXiv:2601.08436, Jan 2026.

9. **PathFinder:** Zhong et al., "PathFinder: Advancing Path Loss
   Prediction for Single-to-Multi-Transmitter Scenario,"
   arXiv:2512.14150, Dec 2025.

10. **RadioUNet:** Levie et al., "RadioUNet: Fast Radio Map Estimation
    with Convolutional Neural Networks," IEEE TWC, 2021.

### Papers que informan las recomendaciones de training

11. **CutMix:** Yun et al., "CutMix: Regularization Strategy to Train
    Strong Classifiers with Localizable Features," arXiv:1905.04899,
    ICCV 2019.

12. **CutMix para segmentacion:** Zhang et al., "Cut to the Mix: Simple
    Data Augmentation Outperforms Elaborate Ones in Limited Organ
    Segmentation Datasets," MICCAI 2024, Paper 674.

13. **Stochastic Depth:** Huang et al., "Deep Networks with Stochastic
    Depth," arXiv:1603.09382, ECCV 2016.

14. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection,"
    arXiv:1708.02002, ICCV 2017.

15. **Adaptive Huber Loss:** Ribeiro et al., "Residual-based Adaptive
    Huber Loss for CQI Prediction in 5G Networks," arXiv:2408.14718, 2024.

16. **Quantization noise:** Widrow & Kollar, "Quantization Noise:
    Roundoff Error in Digital Computation, Signal Processing, Control,
    and Communications," Cambridge University Press, 2008.

17. **EMA dynamics:** "Exponential Moving Average of Weights in Deep
    Learning: Dynamics and Benefits," arXiv:2411.18704, Nov 2024.

18. **NLoS como bottleneck:** Borhani et al., "Millimeter-Wave UAV Channel
    Model with Height-Dependent Path Loss and Shadowing in Urban
    Scenarios," arXiv:2511.10763, Nov 2025.

19. **UNet+ASPP UAV mmWave:** Hussain, "A Multi-Scale Feature Extraction
    and Fusion UNet for Pathloss Prediction in UAV-Assisted mmWave Radio
    Networks," arXiv:2509.09606, Sep 2025.

20. **CKMImageNet:** "CKMImageNet: A Dataset for AI-Based Channel
    Knowledge Map," arXiv:2504.09849, Apr 2025.
