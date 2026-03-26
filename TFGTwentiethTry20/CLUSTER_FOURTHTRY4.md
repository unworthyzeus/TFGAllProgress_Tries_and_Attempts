# FourthTry4 en el cluster (2 GPUs)

## 1. Subir FourthTry4 al cluster

Desde tu máquina local (en `TFGpractice`):

```bash
# Crear directorio remoto
ssh gmoreno@sert.ac.upc.edu "mkdir -p /scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4"

# Subir archivos necesarios
scp TFGFourthTry4/train_cgan.py TFGFourthTry4/config_utils.py TFGFourthTry4/data_utils.py TFGFourthTry4/model_cgan.py TFGFourthTry4/model_unet.py TFGFourthTry4/evaluate_cgan.py TFGFourthTry4/heuristics_cgan.py gmoreno@sert.ac.upc.edu:/scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4/

scp -r TFGFourthTry4/configs TFGFourthTry4/cluster gmoreno@sert.ac.upc.edu:/scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4/

# Copiar dataset si no existe (o enlazar)
ssh gmoreno@sert.ac.upc.edu "test -f /scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4/CKM_Dataset.h5 || cp /scratch/nas/3/gmoreno/TFGpractice/TFGThirdTry3/CKM_Dataset.h5 /scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4/"
```

## 2. Lanzar el job (2 GPUs DDP)

```bash
ssh gmoreno@sert.ac.upc.edu
cd /scratch/nas/3/gmoreno/TFGpractice/TFGFourthTry4
sbatch cluster/run_fourthtry4_2gpu.slurm
```

## 3. Variante: 2 jobs en GPUs distintas

Si quieres lanzar 2 experimentos en paralelo (cada uno en un nodo/GPU distinto):

**Job 1** (2× RTX 2080, partición gpu):
```bash
sbatch --gres=gpu:rtx2080:2 cluster/run_fourthtry4_2gpu.slurm
```

**Job 2** (otra partición, si existe):
```bash
sbatch -p medium_gpu --gres=gpu:2 --export=ALL,CONFIG_PATH=configs/cgan_unet_hdf5_pathloss_hybrid_cuda_max112_blend_db_tinygan_batchnorm_lowresdisc_fourthtry4.yaml,OUTPUT_SUFFIX=ddp2_fourthtry4_medium cluster/run_fourthtry4_2gpu.slurm
```

(Requiere que `run_fourthtry4_2gpu.slurm` no fije `--gres` de forma rígida; si lo hace, crea un segundo script.)

## 4. Monitorear

```bash
squeue -u gmoreno
tail -f logs_train_fourthtry4_ddp_<JOBID>.out
```
