# Guía 4: Simulación de Redes con OMNeT++

Esta guía resume el uso de OMNeT++ en los clústeres de la DAC para simulaciones discretas de red.

## 1. Instalación Personalizada
No se recomienda trabajar con una instalación global del clúster; cada usuario debe tener su propia copia del IDE y binarios.

**Pasos:**
1. Copia tu instalación de OMNeT++ a un directorio de tu home (ej. `~/omnetpp`).
2. Edita el fichero `configure.user` (si vas a compilar) o simplemente asegúrate de que el script de arranque `setenv` apunte a la ruta local.
3. Modifica la variable `IDEDIR` en el script `omnetpp`:
   ```bash
   IDEDIR=$HOME/omnetpp/ide
   ```

## 2. Ejecución Gráfica (IDE)
Para usar el IDE de forma visual desde tu ordenador personal:
1. Conéctate al clúster permitiendo el **X11 Forwarding**:
   ```bash
   ssh -X gmoreno@sert.ac.upc.edu
   ```
2. Carga el entorno de OMNeT++ y lanza el IDE:
   ```bash
   source setenv
   omnetpp
   ```
*Nota: Es deseable que la conexión sea fluida (se recomienda estar en la red de la UPC o tener buena latencia).*

## 3. Ejecución en Lote (Slurm Array Jobs)
Para simulaciones masivas (ej. 100 semillas diferentes), usa los `Array Jobs` de Slurm:
```bash
#!/bin/bash
#SBATCH --job-name=OmnetSim
#SBATCH --array=1-100                 # Ejecuta 100 instancias
#SBATCH --cpus-per-task=1             
#SBATCH --time=02:00:00

# Se ejecuta con Cmdenv (sin entorno gráfico)
opp_run -u Cmdenv -c Config_TFG -r $SLURM_ARRAY_TASK_ID ./mi_simulacion.ini
```

---
**Fuente oficial:**
*   [Uso de OMNeT++ en el Cluster](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/OMNeT++)
