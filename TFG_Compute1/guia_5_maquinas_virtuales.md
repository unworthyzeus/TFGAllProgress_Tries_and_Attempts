# Guía 5: Máquinas Virtuales (KVM)

Esta guía documenta la infraestructura de virtualización disponible en DAC mediante QEMU/KVM.

## 1. Funcionamiento General
El DAC provee infraestructuras de virtualización para casos en los que se requiera el control total de la máquina (**root/sudo**) o entornos aislados que no puedan ejecutarse en el clúster estándar.

## 2. Aceleración de Hardware
Para activar la aceleración de hardware (**KVM**) es necesario tener permisos del grupo `kvm`. Antes de lanzar tu máquina virtual, ejecuta:
```bash
newgrp kvm
```

## 3. Imágenes de SO Base
Existen imágenes preinstaladas disponibles para ser clonadas:
```bash
/Soft/Share/images/     # Carpeta con imágenes de Ubuntu (8.04, 10.04, 12.04)
```

## 4. Mejora de Rendimiento (Scratch 1)
**⚠️ NO ejecutes la máquina virtual directamente desde tu Home.** La lectura/escritura de disco por red ralentizaría tu trabajo y saturaría la infraestructura.

**Método recomendado:**
1. Copia la imagen de la máquina virtual al disco duro local del nodo (**Scratch 1**):
   ```bash
   cp /home/gmoreno/mi_vm.img /scratch/1/gmoreno/mi_vm.img
   ```
2. Ejecuta la máquina desde esa ruta.
3. **Mantenimiento**: Al ser en `/scratch/1/`, recuerda sacar tus resultados al Home o NAS antes de que pasen los 30 días de borrado automático.

---
**Fuente oficial:**
*   [Uso de Máquinas Virtuales en el Cluster](https://www.ac.upc.edu/app/wiki/serveis-tic/Clusters/Users/MaquinaVirtual)
