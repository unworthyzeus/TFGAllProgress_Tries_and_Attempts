#!/bin/bash
#SBATCH --job-name=TrainUNET
#SBATCH --partition=gpu
#SBATCH --qos=small_gpu         # Máximo tiempo y 1 GPU (para no perder prioridad en colas)
#SBATCH --gres=gpu:1            # IMPORTANTE: pedimos 1 tarjeta gráfica
#SBATCH --cpus-per-task=4       # 4 hilos para procesar datos
#SBATCH --mem=32G               # Memoria RAM del host, cuanta más mejor para datasets
#SBATCH --time=11:59:00         # 12 horas limite
#SBATCH --output=unet_%j.log

# 1. Variables y Rutas
USER="gmoreno"
ENV_DIR="/scratch/nas/g/$USER/tfg_env"
DATASET_TAR="/home/$USER/dataset_imagenes_medicas.tar.gz"  
LOCAL_SCRATCH="/scratch/1/$USER/dataset_unet"

echo "=== Arrancando Job ID: $SLURM_JOB_ID en GPU ==="
nvidia-smi

# 2. Cargar Entorno
module purge
module load gcc cuda cudnn
source $ENV_DIR/bin/activate

# 3. Mover Dataset a /scratch/1 (SSD Local ultra-rápido)
echo "Copiando dataset a disco local $LOCAL_SCRATCH..."
mkdir -p $LOCAL_SCRATCH
tar -xzf $DATASET_TAR -C $LOCAL_SCRATCH

# 4. Entrenar la Red
echo "Iniciando script de entrenamiento Python..."
# ATENCION: El script python asume que los checkpoints se guardan en el HOME (persistente)
python train_unet.py --data_path $LOCAL_SCRATCH --epochs 100 --batch_size 16 --save_dir /home/$USER/checkpoints_unet/

# 5. Limpieza (Opcional pero recomendable)
echo "Limpiando disco local..."
rm -rf $LOCAL_SCRATCH

echo "Entrenamiento finalizado."
