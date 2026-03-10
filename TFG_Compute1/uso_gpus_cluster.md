# Guía Completa y Exhaustiva: Uso del Clúster de la UPC (AC/DAC)

Esta guía documenta a fondo el entorno de los clústeres del Departament d'Arquitectura de Computadors basándose en las [Wikis de Serveis TIC y manuales de usuario](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users).  
Permite cubrir no solo la ejecución general, sino herramientas específicas como **TensorFlow**, **OMNeT++**, uso interactivo mediante **VS Code** y gestión de **Nodos/Máquinas Virtuales**.

---

## 0. Primeros Pasos: Cómo Conectarse al Clúster (SSH)
Actualmente, **los detalles exactos de conexión (el nombre de la máquina o de tu clúster) están dentro de la Intranet de UPC**. Sin embargo, de forma general, para conectarte necesitas tres cosas:

1. **Estar en la red de la UPC:** Si estás en tu casa, es obligatorio tener encendida y conectada la VPN de la Universidad (**UPCLink** o FortiClient).
2. **Usuario y Contraseña:** Tal y como has confirmado, tu usuario para esto es **`guillem.moreno.garcia`** y la contraseña es `Mamareentanga54`.
3. **El Servidor (Hostname):** Necesitas saber a qué máquina conectarte. En la ventana de Chrome que hemos abierto, en cuanto entres a cualquiera de esos apartados de la Wiki, te pondrá el nombre del host (suele ser `login.ac.upc.edu` o `sert.ac.upc.edu`, o tal vez te han dado uno especial para tu TFG).

Para conectarte **Ahorita Mismo**, abre una terminal y ejecuta el comando SSH (poniendo el servidor real donde indico HOSTNAME):
```bash
ssh guillem.moreno.garcia@HOSTNAME
# Ejemplo si es el cluster general: ssh guillem.moreno.garcia@login.ac.upc.edu
# (Luego te pedirá la contraseña Mamareentanga54)
```

--- 

## 1. Funcionamiento General (Clúster Sert, CER, etc.)
La arquitectura de los clústeres en DAC se divide entre **Nodos de Login** y **Nodos de Computación (Workers)**, gestionados mediante **SLURM**.

- **El Nodo de Login (Frontend):** Es donde entras inicialmente vía SSH. **No se debe ejecutar ninguna orden pesada aquí** (ni entrenar, ni lanzar simulaciones largas). Su única función es alojar tus ficheros, poder compilar (`make`) e interactuar con la cola para ejecutar con `sbatch`.
- **Los Nodos de Cómputo (Backend):** Máquinas esclavas separadas. Se envían trabajos a estos a través de **Slurm**.
- **Particiones (Colas):** Existen diferentes particiones con límites de tiempo. Por norma general existen `short`, `long` o `gpu`.

### Envío de un Trabajo (Batch Job)
Crea un archivo llamado `job.sh` y ejecútalo con `sbatch job.sh`.
```bash
#!/bin/bash
#SBATCH --job-name=mi_tarea
#SBATCH --partition=gpu
#SBATCH --output=logs_%j.out
#SBATCH --error=errores_%j.err
#SBATCH --time=12:00:00           # Tiempo maximo de ejecucion
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # Hilos de procesador (cores)
#SBATCH --mem=8G                  # Memoria RAM a reservar

# Instrucciones
echo "Ejecutando en el nodo:" $SLURMD_NODENAME
python mi_codigo.py
```

---

## 2. Uso Avanzado de GPUs (`UsGPUs`)
Para pedir una gráfica no basta con la partición `gpu`, debes especificar explícitamente el uso del recurso (`gres`).

### Solicitud de hardware
Añade a tu script la siguiente directiva:
```bash
#SBATCH --gres=gpu:1         # Pide 1 tarjeta GPU generica
#SBATCH --gres=gpu:rtx3080:1 # Pide especificamente el modelo RTX 3080 (dependiendo de si la etiqueta exacta del clúster lo soporta)
```
*Si necesitas más de una GPU para modelos distribuidos, puedes pedir `gpu:2`, pero ten en cuenta que pasarás mucho más tiempo en cola esperando a que haya 2 libres en la misma máquina.*

### Verificación del Entorno
Al arrancar, si lanzas un trabajo con gráfica, asegúrate de comprobar que CUDA te la ha reconocido:
```bash
srun --partition=gpu --gres=gpu:1 --pty bash  # Abre una sesion consola en una GPU libre
nvidia-smi                                    # Verifica las graficas conectadas
```

---

## 3. Uso de TensorFlow / PyTorch (`TensorFlow`)
Los entornos de Deep Learning sufren un problema común: si no los configuras bien, monopolizan toda la memoria (VRAM) de la tarjeta, limitando a otros usuarios incluso si tu red neuronal no la consume.

### Carga de dependencias
El clúster usa `lmod` (Modules) para instalar versiones. Comprueba los módulos disponibles de CUDA y Python:
```bash
module avail
module load gcc
module load cuda/11.x  # Cambiar por la version instalada más reciente
module load cudnn/8.x
```

### Entorno aislado (Conda)
La forma más limpia de utilizar TensorFlow o PyTorch es tener tu propio Anaconda (o Miniconda) en la carpeta de tu usuario.
```bash
# Primero lo activas
source ~/miniconda3/bin/activate
conda activate tf_env
```

### 🔴 Importante para TensorFlow
Para evitar que monopolices gráficamente la tarjeta (bloqueando clústeres compartidos), añade esta variable de entorno antes de ejecutar TensorFlow **obligatoriamente**:
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
# o en codigo python:
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

---

## 4. Simulaciones con OMNeT++ (`OMNeT++`)
Si para la investigación haces simulación discreta de redes usando OMNeT++, se suele requerir ejecutar miles de semillas estadísticamente independientes. En lugar de hacer mil envíos, se usan los **Slurm Array Jobs**.

**Ejemplo de Job de OMNeT++ masivo (`array_omnet.sh`)**:
```bash
#!/bin/bash
#SBATCH --job-name=OmnetSim
#SBATCH --array=1-100                 # Ejecuta 100 instancias!
#SBATCH --cpus-per-task=1             # OMNeT++ por defecto es 1 solo hilo si no se paralelizacion MPI
#SBATCH --time=02:00:00

# El valor $SLURM_ARRAY_TASK_ID cambiara en cada ejecucion desde 1 a 100
echo "Arrancando semilla número $SLURM_ARRAY_TASK_ID"

# Se ejecuta siempre desde linea de comandos con entorno Cmdenv
opp_run -u Cmdenv -c MiConfiguracion -r $SLURM_ARRAY_TASK_ID ./mi_simulador.ini
```
*Compila siempre previamente (`make clean && make`) interactuando en el nodo de login antes de mandar los "sbatch".*

---

## 5. Desarrollo Integrado Interactivo: Visual Studio Code (`vscode`)
En lugar de subir código con FTP, puedes editar y probar tus modelos usando el editor VS Code instalado en tu ordenador directamente alojado en la UPC.

1. **Instalación:** Necesitas la extensión **`Remote - SSH`** de Microsoft en tu VS Code local.
2. **Conexión al Clúster:** Aprieta `F1` y selecciona `Remote-SSH: Connect to Host...`. Pon el servidor por defecto (ej. `<usuario>@login.ac.upc.edu`).
3. **El Archivo de Configuración:** Suele ser cómodo crear el fichero `~/.ssh/config` en tu casa con estos datos:
   ```ssh
   Host UPC-Cluster
       HostName login.ac.upc.edu
       User tunombre.deusuario
       # Si hace falta estar conectado a UPCLink (VPN) o hacer tunel a través de una puerta de enlace:
       # ProxyJump servidor.puerta.upc.edu
   ```
4. **Cuidado con los Nodos de Login:** VS Code instala "extensiones de servidor" en el propio PC remoto. Si lanzas scripts pesados desde la terminal integrada de VS Code ¡estarás parando el servidor de Login a todos! Cualquier ejecución hazla con un `sbatch` o usa `srun --pty bash` dentro de ese terminal.

---

## 6. Recursos Asignados: Máquinas Virtuales (`MaquinaVirtual`)
A diferencia de los clústeres puramente compartidos (Slurm), existen proyectos, TFGs o laboratorios en los que el Departament AC da de alta una **Máquina Virtual (VM) completa** en su infraestructura interna (habitualmente VMware, KVM/QEMU u OpenNebula).

*   **Autonomía y Root:** Si te asignan una VM para tu TFG, te entregarán acceso SSH a una IP interna. Al revés que en `Sert`, aquí tendrás **privilegios de `sudo` (root)** absolutos sobre ella, lo que te permite instalar librerías a nivel de sistema (`apt install`), usar contenedores Docker/Podman de forma nativa e instalar cualquier dependencia.
*   **Recursos fijos:** Los recursos (CPUs, Memoria y/o GPUs Virtualizadas [vGPU]) son estáticos e indicados a la hora de contratar o crear la máquina. No compites con otros mediante colas: lo que la VM tiene, lo tienes tú en exclusiva las 24horas.
*   **Acceso:** Si la VM posee un rango de IPs internas del recinto, necesitarás conectarte a través del servicio de VPN (UPCLink o Forticlient) de la universidad obligatoriamente.

---
> Para más detalles, comandos del sistema, listas exactas de cuotas VRAM, y soporte directo del personal del DAC, consultar las rutas web originales en la Wiki oficial **[Serveis TIC -> Clusters -> Users](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters)**.
