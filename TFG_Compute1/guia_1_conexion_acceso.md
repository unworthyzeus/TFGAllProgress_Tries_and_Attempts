# Guía 1: Conexión y Acceso al Clúster (DAC)

Esta guía detalla cómo acceder a los recursos de computación del Departament d'Arquitectura de Computadors (DAC) de la UPC.

## 1. Requisitos Previos
*   **VPN**: Si accedes desde fuera del campus, debes conectar la VPN de la UPC (**UPCLink**).
*   **Credenciales**:
    *   **Usuario Cluster/SSH**: `gmoreno`
    *   **Usuario Wiki/SSO**: `guillem.moreno.garcia`

## 2. Acceso vía SSH
Para entrar al nodo de entrada (frontend):
```bash
ssh gmoreno@sert.ac.upc.edu
# O también:
ssh gmoreno@login.ac.upc.edu
```
*Nota: El alias `sert` suele redirigir automáticamente a uno de los nodos de entrada disponibles (ej. `sert-entry-1`).*

### ⚠️ Puntos a tener en cuenta en tu primera conexión:
Aunque tenemos mucha información, los clústeres pueden cambiar sus políticas de acceso por seguridad. La primera vez que te conectes, ten en mente lo siguiente:
1.  **¿Te pide Contraseña o Clave Pública (SSH Key)?:** La mayoría de universidades permiten acceder la primera vez con la contraseña (`Mamareentanga54`), pero algunos servicios modernos del DAC podrían requerir que generes una clave SSH en tu ordenador y la subas a la Intranet previamente. Si el comando de arriba rechaza tu contraseña repetidas veces, busca "SSH Keys" en tu perfil de la Intranet.
2.  **Gateways (Nodos Bastión):** `sert.ac.upc.edu` o `login.ac.upc.edu` actúan ya como servidores de salto (pasarelas). Esto significa que **nunca** puedes conectar directamente desde Internet al nodo GPU `sert-2001`. Siempre hay que usar SSH hacia el login primero.
3.  **El Formato del Usuario:** El clúster reconoce tu identidad local del DAC (`gmoreno`). Tu correo general (`guillem.moreno.garcia`) es únicamente para entrar a la página web (SSO) y validarte allí. Para la consola Linux, tú eres `gmoreno`.

## 3. Almacenamiento y Cuotas
Existen tres tipos de directorios principales:

| Tipo | Ruta | Características |
| :--- | :--- | :--- |
| **Home** | `/home/gmoreno` | Cuota limitada. Tiene **Backup**. |
| **Scratch Local** | `/scratch/1/gmoreno` | Disco rápido en el nodo. **Se borra cada 30 días**. Sin backup. |
| **NAS Scratch** | `/scratch/nas/g/gmoreno` | Accesible desde todos los nodos. Recomendado para entornos `venv`. |

*Importante: Usa el NAS Scratch para instalar librerías pesadas y evitar llenar tu Home.*

## 4. Ejecución Interactiva (Slurm)
Para obtener una consola en un nodo de cálculo y probar comandos en tiempo real:
```bash
srun -A interactive -p interactive --pty /bin/bash
```

## 5. Visual Studio Code (Remote SSH)
**⚠️ NO conectes VS Code directamente a los nodos de login.** El servidor de VS Code consume muchos recursos y el sistema matará tus procesos.

**Método correcto:**
1. Conéctate por SSH normal al clúster.
2. Ejecuta el script de túnel SSH proporcionado en la Wiki (sección vscode).
3. Conecta tu VS Code local al puerto que te indique el script.

---
**Fuentes oficiales:**
*   [Sert - Funcionament General](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Sert/FuncionamentGeneral)
*   [Uso de VSCode en el Cluster](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/vscode)
