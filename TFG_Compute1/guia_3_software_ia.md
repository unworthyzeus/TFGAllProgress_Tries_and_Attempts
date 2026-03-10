# Guía 3: Deep Learning con TensorFlow y PyTorch

Esta guía detalla la configuración de entornos de Inteligencia Artificial para el clúster DAC de la UPC.

## 1. Instalación Recomendada
Instala tus entornos en el **NAS scratch** para evitar errores de cuota en tu Home.

### Pasos para crear el entorno:
1. Accede a un nodo de cálculo de forma interactiva (para no cargar el login):
    ```bash
    srun -A interactive -p interactive --pty /bin/bash
    ```
2. Crea el directorio y el entorno virtual:
    ```bash
    mkdir -p /scratch/nas/g/gmoreno/tf_env
    python3 -m venv /scratch/nas/g/gmoreno/tf_env
    ```
3. Activa el entorno e instala:
    ```bash
    source /scratch/nas/g/gmoreno/tf_env/bin/activate
    pip install tensorflow
    # O para PyTorch
    pip install torch torchvision
    ```

## 2. Gestión de Memoria VRAM (CRÍTICO para cGANs y UNETs)
Las arquitecturas **cGAN** (que requieren dos redes compitiendo: Generador y Discriminador) y **UNET** (que suele manejar imágenes de alta resolución) consumen mucha VRAM.

**Para evitar fallos de "Out of Memory" (OOM):**
*   **Memory Growth**: Añade esta variable antes de ejecutar tu script para que TensorFlow solo pida lo que necesita:
    ```bash
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    ```
*   **Batch Size**: Si tienes un error de memoria, reduce el `batch_size` en tu código. Es preferible un batch pequeño que no poder entrenar.
*   **Mixed Precision**: Si usas PyTorch o TF 2.x, activa la precisión mixta (`float16`) para reducir a la mitad el uso de VRAM sin perder calidad significativa.

## 3. Optimización del Entrenamiento: Dataset en Scratch
Entrenar una UNET implica leer miles de imágenes. Si las dejas en tu Home, el entrenamiento irá lento por el cuello de botella de la red (NAS).

**Flujo de trabajo recomendado para cGAN/UNET:**
1. En tu script de Slurm (`sbatch`), copia tu dataset comprimido al disco local del nodo:
   ```bash
   cp /home/gmoreno/mi_dataset.tar.gz /scratch/1/gmoreno/
   tar -xzf /scratch/1/gmoreno/mi_dataset.tar.gz -C /scratch/1/gmoreno/
   ```
2. Entrena apuntando a `/scratch/1/gmoreno/mi_dataset`.
3. **Checkpoints**: Asegúrate de que las rutas de guardado de los modelos (`.h5`, `.pth`, `.ckpt`) apunten a tu **Home** o **NAS Scratch**, ya que si el nodo falla o termina el tiempo de Slurm, lo que esté en `/scratch/1` se perderá.

## 4. Carga de Módulos (Cuda/Cudnn)
El clúster dispone de versiones optimizadas.
```bash
module load gcc
module load cuda/11.x  # Recomendada para compatibilidad con las RTX 2080/3080
module load cudnn
```

---
**Fuente oficial:**
*   [Uso de TensorFlow en el Cluster](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/TensorFlow)
*   [Hardware y GPUs del DAC](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/UsGPUs)
