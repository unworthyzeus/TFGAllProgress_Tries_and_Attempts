import os
import sys

# 1. EVITAR que TF consuma toda la VRAM del nodo entero
# Esto es esencial en el cluster del DAC, de lo contrario bloquearás la gráfica para otros usuarios
# y te darán "Out Of Memory" si compartes máquina
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

def test_gpu_allocation():
    print("Verificando dispositivos físicos...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # 2. Configuración manual de memory growth por código (redudante pero más seguro)
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Detectadas GPUs: {len(gpus)} Físicas, {len(logical_gpus)} Lógicas")
        except RuntimeError as e:
            # Los dispositivos lógicos no pueden modificarse tras inicializar la gráfica
            print(e)
    else:
        print("ADVERTENCIA: No se han detectado GPUs de NVIDIA. Estás entrenando en CPU!")

if __name__ == "__main__":
    test_gpu_allocation()
    print("\n--- ¡Todo listo para construir tu Modelo cGAN o UNET! ---")
    
    # Aquí iría el código de tu arquitectura...
    # Generador = tf.keras.Sequential(...)
    # Discriminador = tf.keras.Sequential(...)
