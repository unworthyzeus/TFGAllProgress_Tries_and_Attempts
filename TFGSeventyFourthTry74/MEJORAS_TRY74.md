# Mejoras Propuestas Para Try 74

Fecha: 2026-04-17

## Estado actual

El `Try 74` actual ya no parece estar roto por inestabilidad de optimizacion como pasaba en versiones anteriores. Ahora mismo el patron es:

- el modelo aprende bastante bien `LoS`
- el modelo sigue fallando mucho en `NLoS`
- el cuello de botella parece ser representacion/modelado de sombra y bloqueo, no tanto "mas capacidad" del backbone

Referencia principal:

- `cluster_outputs/TFGSeventyFourthTry74/try74_expert_band4555_allcity/validate_metrics_latest.json`

Metricas observadas:

- `overall RMSE = 27.46 dB`
- `LoS RMSE = 5.99 dB`
- `NLoS RMSE = 42.90 dB`
- `epoch = 8`
- `best_epoch = 8`

Interpretacion:

- el modelo ya captura el campo suave/global
- la parte dificil sigue siendo la penalizacion extra por bloqueo, diffraction y sombra
- por tanto, ahora interesa priorizar ideas que ataquen especificamente el error NLoS

## Diagnostico

La lectura mas probable es:

1. `PMHNet` ya esta haciendo lo razonable para la parte geometrica suave.
2. Los canales extra actuales (`distance`, `LoS`, `tx_depth`, `elevation`, `building_mask`, obstruction features) ayudan, pero no fuerzan suficientemente una representacion explicita del "excess loss" NLoS.
3. El objetivo actual sigue permitiendo que el modelo gane mucho mejorando LoS y solo parcialmente NLoS.

En otras palabras:

- antes el problema era de configuracion/objetivo
- ahora el problema parece ser de representacion y priorizacion del error dificil

## Mejoras priorizadas

### 1. Dejar correr este Try 74 hasta ver el primer escalon de LR

No conviene cortarlo demasiado pronto.

Motivo:

- el run actual va bajando
- aun no ha dado tiempo a ver si el scheduler ayuda de verdad en la parte NLoS

Regla practica:

- esperar al menos a un primer `LR drop`
- reevaluar si `NLoS RMSE` sigue practicamente clavado por encima de `~40 dB`

### 2. Siguiente experimento de bajo riesgo: `sample_rmse_reweight`

Esta es mi siguiente prueba favorita porque ya existe en el trainer y no requiere cirugia grande.

Soporte en codigo:

- `train_partitioned_pathloss_expert.py` ya crea `SampleRMSEBuffer`
- ya multiplica `g_loss` por dificultad de muestra

Ventaja:

- prioriza escenas realmente dificiles
- es mas fino que solo subir `nlos_reweight_factor`
- puede empujar mejor las muestras con mucha sombra sin destrozar la parte LoS

Hipotesis:

- si el problema real esta concentrado en unas escenas NLoS duras, esto deberia ayudar mas que seguir subiendo pesos por pixel

### 3. Cambiar `nlos_focus_loss` de `rmse` a `hard_huber`

Esto tambien es de bajo riesgo porque el modo ya existe en el trainer.

Motivo:

- el error NLoS tiene pinta de tener cola pesada
- RMSE pura deja que unos pocos outliers dominen demasiado
- `hard_huber` puede mantener presion sobre NLoS sin quedar secuestrado por unos pocos agujeros negros

Propuesta inicial:

- `nlos_focus_loss.mode = hard_huber`
- `huber_delta = 6.0 dB` como punto de partida
- ajustar `alpha/gamma` solo si hace falta despues

### 4. Confirmar/precomputar obstruction features

Ahora mismo el YAML apunta a:

- `../TFGFortySeventhTry47/precomputed/obstruction_features_u8.h5`

Pero localmente ese fichero no existe.

Implicacion:

- hay riesgo de que los obstruction features se esten calculando on-the-fly
- eso puede meter latencia, variabilidad y overhead innecesario

Accion recomendada:

- comprobar en cluster si el fichero existe de verdad
- si no existe, generarlo/precomputarlo y dejarlo fijo

### 5. Try 75: no depender solo de FiLM para la altura

Para `Try 75`, mi recomendacion es no usar solo condicionamiento global via FiLM.

Mejor idea:

- mantener FiLM
- pero ademas meter la altura como representacion espacial/canal

Opciones:

- plano constante `h_tx_norm`
- plano constante `log(h_tx)`
- pequeno banco RBF/Fourier de altura convertido a canales

Motivo:

- papers recientes de representacion de features sugieren que meter features escalares como canales mejora la generalizacion frente a usarlas solo como escalares/globales

## Mejoras mas ambiciosas

### 6. Cabeza auxiliar de bloqueo / sombra

No una cabeza auxiliar cualquiera, sino una que fuerce al modelo a aprender estructura NLoS.

Opciones razonables:

- cabeza auxiliar `LoS/NLoS`
- cabeza auxiliar `shadow depth`
- cabeza auxiliar `excess loss over smooth field`

Motivo:

- ahora mismo el modelo aprende solo el mapa final
- no se le obliga a factorizar explicitamente el fenomeno fisico que mas duele

### 7. Descomposicion aprendida: `PL = base_suave + shadow_residual`

Esto puede ser la idea mas potente sin volver al prior analitico.

Concepto:

- una rama aprende el campo suave/global
- otra rama aprende la penalizacion por bloqueo/sombra
- la salida final es la suma de ambas

Ventaja:

- obliga a separar lo facil de lo dificil
- esta muy alineado con como pensamos fisicamente el problema
- permite atacar NLoS como exceso de perdida, no como todo el mapa de golpe

### 8. Curriculum de altura para Try 75

Si `Try 74` acaba siendo un buen pretraining a `45-55 m`, la continuacion natural no deberia ser saltar directamente a toda la distribucion sin pasos intermedios.

Propuesta:

1. `45-55 m`
2. `35-70 m`
3. `25-100 m`
4. `full range`

Motivo:

- permite que el modelo conserve la estructura aprendida en altura casi fija
- hace mas suave el aprendizaje de la deformacion con altura

## Cosas que ahora mismo no venderia como "siguiente palanca principal"

### 1. Reescribir el backbone para hacerlo "mas PMNet"

No es mi apuesta principal.

Motivo:

- el run actual ya parece suficientemente PMNet-like
- el patron de error no sugiere falta de backbone generalista
- sugiere falta de modelado explicito de sombra/NLoS

### 2. `corridor_weighting` como solucion inmediata

Ojo: en este branch de `Try 74`, `corridor_weighting` aparece en YAML/generador, pero no parece estar realmente integrado en el trainer actual.

Conclusión:

- no contarlo como knob real hasta cablearlo de verdad

### 3. `dual_los_nlos_head` como si ya estuviera operativo

Igual que arriba:

- aparece en generacion de config
- pero no he visto uso real en este trainer

Conclusión:

- si se quiere probar, primero hay que implementarlo de verdad

## Nota de sanidad sobre metricas

En el JSON actual aparece `path_loss_513 = Infinity` mientras el `path_loss` principal es finito.

Esto sugiere que:

- hay algun problema en esa ruta de evaluacion/resumen
- no conviene usar `path_loss_513` para decidir nada hasta revisarlo

Accion recomendada:

- auditar la ruta de metricas full-res antes de tomar decisiones con ese campo

## Orden recomendado de siguientes pasos

1. Dejar correr el `Try 74` actual hasta ver al menos un escalon de LR.
2. Si `NLoS` sigue clavado, lanzar un `Try 74b` con:
   - `sample_rmse_reweight.enabled = true`
   - `nlos_focus_loss.mode = hard_huber`
3. Confirmar/precomputar obstruction features en cluster.
4. Diseñar `Try 75` con altura por doble via:
   - FiLM global
   - canales espaciales de altura
5. Si aun asi NLoS sigue muy alto, implementar la descomposicion:
   - `base_suave + shadow_residual`

## Referencias externas usadas para orientar estas propuestas

- PMNet: Robust Pathloss Map Prediction via Supervised Learning
- A Scalable and Generalizable Pathloss Map Prediction
- RadioUNet
- Investigating Map-Based Path Loss Models: A Study of Feature Representations in Convolutional Neural Networks
- UAV-aided Radio Map Construction Exploiting Environment Semantics
- Effective outdoor pathloss prediction: A multi-layer segmentation approach with weighting map
