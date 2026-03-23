"""Rutas de datasets para subida al cluster (altura UAV en HDF5, no CSV)."""
from __future__ import annotations

import os
from pathlib import Path


def prepare_ssh_transport_for_large_uploads(ssh_client) -> None:
    """Llamar justo después de connect() y antes de open_sftp() para subidas HDF5 grandes."""
    try:
        t = ssh_client.get_transport()
        if t is None:
            return
        # Ventanas pequeñas por defecto → a veces SFTP devuelve status 4 "Failure" a mitad de transferencia.
        if hasattr(t, "default_window_size"):
            t.default_window_size = max(int(getattr(t, "default_window_size", 0)), 268_435_456)
        if hasattr(t, "default_max_packet_size"):
            t.default_max_packet_size = max(int(getattr(t, "default_max_packet_size", 0)), 32_768)
        pkt = getattr(t, "packetizer", None)
        if pkt is not None:
            pkt.REKEY_BYTES = max(getattr(pkt, "REKEY_BYTES", 0), 2**40)
            pkt.REKEY_PACKETS = max(getattr(pkt, "REKEY_PACKETS", 0), 2**40)
    except Exception:
        pass

# Nombre canónico en el cluster (un solo fichero para alturas + mapas si es el .h5 combinado)
REMOTE_ANTENNA_H5_NAME = "CKM_180326_antenna_height.h5"
LOCAL_ANTENNA_H5_CANDIDATES = (
    "CKM_180326_antenna_height.h5",
    "CKM_Dataset_180326_antenna_height.h5",
)
REMOTE_MAIN_MAPS_H5_NAME = "CKM_Dataset_180326.h5"


def datasets_dir(practice_root: Path) -> Path:
    return practice_root / "Datasets"


def resolve_antenna_height_h5(practice_root: Path) -> Path | None:
    """Primer .h5 local que exista (nombre corto o estilo CKM_Dataset_*)."""
    d = datasets_dir(practice_root)
    for name in LOCAL_ANTENNA_H5_CANDIDATES:
        p = d / name
        if p.is_file():
            return p
    return None


def resolve_main_maps_h5(practice_root: Path) -> Path | None:
    p = datasets_dir(practice_root) / REMOTE_MAIN_MAPS_H5_NAME
    return p if p.is_file() else None


def remote_antenna_h5_path(remote_base: str) -> str:
    return f"{remote_base.rstrip('/')}/Datasets/{REMOTE_ANTENNA_H5_NAME}"


def remote_main_h5_path(remote_base: str) -> str:
    return f"{remote_base.rstrip('/')}/Datasets/{REMOTE_MAIN_MAPS_H5_NAME}"


def upload_if_missing_file(sftp, mkdir_p, local_path: Path, remote_path: str) -> None:
    """Sube un fichero si falta en remoto o el remoto está vacío / distinto tamaño al local."""
    local_size = local_path.stat().st_size
    if local_size == 0:
        print(f"Skip upload (local file is 0 bytes): {local_path}")
        return

    try:
        st = sftp.stat(remote_path)
        if st.st_size == local_size:
            print(f"Skip (exists, {local_size} bytes): {remote_path}")
            return
        print(
            f"Remote size mismatch or stale file (remote {st.st_size} B vs local {local_size} B); "
            f"replacing {remote_path!r}..."
        )
        try:
            sftp.remove(remote_path)
        except OSError:
            pass
    except FileNotFoundError:
        pass

    mkdir_p(sftp, os.path.dirname(remote_path).replace("\\", "/"))
    gib = local_size / (1024.0**3)
    print(f"Upload {local_path.name} ({gib:.2f} GiB) -> {remote_path}")
    try:
        # putfo + confirm=False: menos problemas que put() en Windows/ficheros grandes.
        with open(local_path, "rb") as fh:
            sftp.putfo(fh, remote_path, local_size, confirm=False)
    except OSError as e:
        raise RuntimeError(
            f"SFTP falló al subir a {remote_path!r}: {e!r}. "
            "Ejecuta `quota -s` en el cluster: si scratch supera la cuota, libera ~1–2 GiB "
            "(outputs viejos, checkpoints, duplicados) antes de subir. "
            f"Borra en servidor cualquier .h5 a 0 B. Alternativa: scp/rsync del .h5."
        ) from e


def mkdir_p_with_quota_hint(sftp, remote_path: str) -> None:
    """mkdir -p; si falla (p. ej. cuota scratch llena), mensaje explícito."""
    parts = [p for p in remote_path.strip("/").split("/") if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            try:
                sftp.mkdir(current)
            except OSError as e:
                raise RuntimeError(
                    f"No se pudo crear {current!r}: {e!r}. "
                    "En SERT suele ser cuota de scratch superada (`quota -s`). "
                    "Libera espacio (rm outputs/checkpoints viejos bajo TFGpractice) y reintenta."
                ) from e
