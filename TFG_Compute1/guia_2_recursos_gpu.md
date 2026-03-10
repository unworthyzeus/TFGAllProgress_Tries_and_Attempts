# Guía 2: Recursos de GPU y Gestión de Colas (SLURM)

Esta guía documenta cómo solicitar, verificar y gestionar las tarjetas gráficas del clúster DAC.

## 1. Concepto de Recursos de GPU
En el clúster de la DAC, las gráficas se gestionan como recursos genéricos (`Gres`). No basta con la partición `gpu`, debes solicitar la gráfica explícitamente.

## 2. Cómo solicitar una o más GPUs
Añade a tu script de envío (`sbatch`) la siguiente directiva:
```bash
#SBATCH --gres=gpu:1           # Pide 1 tarjeta GPU genérica
```
Si necesitas trabajar con modelos que usen múltiples tarjetas, puedes pedir:
```bash
#SBATCH --gres=gpu:2           # Pide 2 GPUs
```
*Se recomienda pedir solo lo necesario para acortar el tiempo en cola.*

## 3. Particiones y Límites de Tiempo para GPU
El acceso a las GPUs está regulado por la Calidad de Servicio (QoS). Dependiendo de cuántas necesites para tu **cGAN** o **UNET**, deberás usar una configuración u otra:

| QoS / Partición | Nº GPUs | Uso Típico | Límite Tiempo (aprox) |
| :--- | :--- | :--- | :--- |
| **`small_gpu`** | 1 | Entrenamiento estándar / Debug | 4 - 12 horas |
| **`medium_gpu`** | 2 - 4 | Modelos grandes / Paralelismo | 12 - 24 horas |
| **`big_gpu`** | 5 - 8 | Entrenamiento masivo (Multinode) | Variable (consultar `sinfo`) |

*Tip: Para tus UNETs, si no usas entrenamiento distribuido, quédate en `small_gpu` para entrar antes en la cola.*

## 4. Hardware Detallado: El Nodo GPU `sert-2001`
El principal recurso de cómputo gráfico para usuarios generales/TFG es el nodo **`sert-2001`**.

*   **Configuración**: 
    *   **8x NVIDIA RTX 2080 Ti** (Cada una con **11 GB** de VRAM GDDR6).
    *   CPU: Intel Xeon Silver 4210R.
    *   RAM del Sistema: 128 GB.
*   **Otros Modelos**: También existen nodos con **NVIDIA RTX 3080** (más potencia de cálculo pero similar o mayor VRAM según modelo).

### Cómo filtrar por modelo en tu script de Slurm:
Si tu código de IA está optimizado para una arquitectura concreta, puedes pedirla así:
```bash
#SBATCH --gres=gpu:rtx2080ti:1   # Pide específicamente una 2080 Ti
#SBATCH --gres=gpu:rtx3080:1     # Pide específicamente una 3080
```

## 5. Probar la GPU de forma interactiva (Debug de cGAN/UNET)
Antes de lanzar un entrenamiento de 10 horas, prueba que tu código detecta la GPU:
```bash
# Solicita una sesión de 1 hora con 1 GPU
srun -A gpu -p gpu --gres=gpu:1 --time=01:00:00 --pty /bin/bash
# Una vez dentro:
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 6. Comandos de Inspección Avanzada
*   **`sinfo -N -l -p gpu`**: Muestra qué nodos de la cola GPU están "idle" (libres).
*   **`scontrol show node sert-2001`**: Te dirá exactamente cuántas GPUs están en uso ahora mismo en ese nodo.

---
**Fuente oficial:**
*   [Uso de GPUs en el Cluster](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/UsGPUs)
*   [Características del Cluster Sert](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Sert/Caracteristiques)
