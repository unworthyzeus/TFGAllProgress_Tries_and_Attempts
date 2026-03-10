# TFG Compute 1: Guía y Scripts para el Clúster de la UPC (DAC)

Este directorio contiene todo el material técnico, guías de uso y plantillas de código para conectarse, configurar y entrenar modelos de Inteligencia Artificial (cGANs, UNETs) en el clúster del **Departament d'Arquitectura de Computadors (DAC)** de la UPC.

Toda la información ha sido extraída y contrastada con la [Wiki oficial de Serveis TIC](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters) asumiendo el usuario genérico `gmoreno`.

---

##  1. Guías Teóricas y de Configuración
Estas guías en formato Markdown detallan paso a paso cómo gestionar el clúster:

*   [`guia_1_conexion_acceso.md`](./guia_1_conexion_acceso.md): Explica cómo conectarte por SSH (`sert.ac.upc.edu`), qué credenciales usar, el requerimiento de la VPN de la UPC y dónde almacenar tus archivos (Home vs Scratch NAS vs Scratch Local).
*   [`guia_2_recursos_gpu.md`](./guia_2_recursos_gpu.md): Detalla el uso de **SLURM**. Explica cómo pedir gráficas (ej. RTX 2080 Ti de 11GB del nodo `sert-2001`), cómo interactuar con ellas en tiempo real y qué colas (QoS) usar según te hagan falta 1 o más GPUs.
*   [`guia_3_software_ia.md`](./guia_3_software_ia.md): Probablemente **la más importante**. Contiene las reglas de oro para entrenar redes grandes (cGANs y UNETs) en relación a evitar el bloqueo de memoria (OOM) en TensorFlow/PyTorch y copiar tu dataset al SSD local para más velocidad.
*   [`guia_4_simulacion_redes.md`](./guia_4_simulacion_redes.md): Explicación breve sobre cómo correr simulaciones masivas (Array Jobs) utilizando OMNeT++.
*   [`guia_5_maquinas_virtuales.md`](./guia_5_maquinas_virtuales.md): Alternativa al clúster compartido: cómo usar QEMU/KVM en la infraestructura del DAC si necesitas privilegios de root (`sudo`).
*   [`uso_gpus_cluster.md`](./uso_gpus_cluster.md): Documento inicial general con apuntes rápidos de SLURM e inventario de GPUs (funciona como resumen rápido).

---

##  2. Scripts Prácticos (Carpeta `scripts_slurm/`)
Dentro de la carpeta `scripts_slurm` encontrarás plantillas de código directamente ejecutables en el clúster para que no empieces de cero:

*   [`01_setup_env.sh`](./scripts_slurm/01_setup_env.sh): **Script Inicial.** Ejecútalo en modo interactivo (`srun --pty bash`) la primera vez que entres. Te crea un Entorno Virtual aislado en el disco NAS (no en el Home) y te instala TensorFlow, PyTorch, OpenCV, etc. cargando el módulo de CUDA 11.
*   [`02_train_unet_dataset_scratch.sh`](./scripts_slurm/02_train_unet_dataset_scratch.sh): **Plantilla Maestra de SLURM.** Este es el fichero que mandarás ejecutar con `sbatch 02_train_unet_dataset_scratch.sh`. Muestra la práctica recomendada: pide 1 sola GPU, 12 horas de límite de tiempo, descomprime tu dataset super rápido en el disco de la máquina (`/scratch/1`) y lanza tu entrenamiento allí.
*   [`03_base_train_tf_vram.py`](./scripts_slurm/03_base_train_tf_vram.py): **Plantilla Python.** Esqueleto base de Python que debes incluir en todos tus entrenamientos con TensorFlow. Activa explícitamente el "Memory Growth" en tu `rtx2080ti` para no bloquear el clúster a tus compañeros del DAC.

---
**Recuerda el flujo de trabajo final:** Conéctate a la VPN > Entra por terminal SSH > Manda tu `sbatch` para entrenar > ¡Déjalo ejecutando toda la noche!
