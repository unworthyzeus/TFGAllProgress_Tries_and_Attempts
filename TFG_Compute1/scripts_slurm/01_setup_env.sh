#!/bin/bash
# Script para configurar el entorno de IA en el cluster DAC por primera vez.
# Ejecuta esto DESDE UN NODO INTERACTIVO, no dejes que corra en el nodo de login.
# srun -A interactive -p interactive --pty /bin/bash

# 1. Definir variables
USER="gmoreno"
ENV_DIR="/scratch/nas/g/$USER/tfg_env"

echo "=== Iniciando configuración del entorno ==="

# 2. Cargar módulos necesarios
module purge
module load gcc
module load cuda/11.8  # Ajusta a la versión disponible más reciente que requieras
module load cudnn/8.6

# 3. Crear el entorno virtual en el NAS (para no bloquear la cuota del Home)
echo "Creando entorno virtual en $ENV_DIR..."
mkdir -p /scratch/nas/g/$USER
python3 -m venv $ENV_DIR

# 4. Activar e instalar dependencias
echo "Instalando dependencias pesadas..."
source $ENV_DIR/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar PyTorch (ajusta la versión de CUDA según corresponda)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar TensorFlow (si prefieres TF)
pip install tensorflow

# Instalar librerías complementarias típicas para UNETs / cGANs
pip install numpy pandas matplotlib scikit-image opencv-python tqdm albumentations

echo "=== Entorno configurado correctamente ==="
echo "Para usarlo en el futuro, simplemente ejecuta:"
echo "source $ENV_DIR/bin/activate"
