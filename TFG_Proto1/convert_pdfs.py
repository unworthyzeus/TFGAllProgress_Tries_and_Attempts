"""
Convert all PDFs in docs/ to Markdown using Marker.
Optimizado para RTX 3050 Ti 4GB: CUDA + control estricto de VRAM.
"""

import gc
import io
import os
import re
import sys
import hashlib
import shutil
from pathlib import Path

import pypdfium2 as pdfium
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Configuración antes de cargar modelos
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MODEL_DTYPE"] = "fp16"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LAYOUT_BATCH_SIZE"] = "1"
os.environ["RECOGNITION_BATCH_SIZE"] = "1"
os.environ["DETECTOR_BATCH_SIZE"] = "1"
os.environ["ORDER_BATCH_SIZE"] = "1"


def sanitize_stem(name: str, max_len: int = 80) -> str:
    clean = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", name)
    clean = re.sub(r"\s+", " ", clean).strip(" .")
    if len(clean) <= max_len:
        return clean
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:10]
    return f"{clean[:max_len-11]}_{digest}"


def to_image_bytes(img_name: str, img_data):
    if isinstance(img_data, (bytes, bytearray, memoryview)):
        return img_name, bytes(img_data)

    if hasattr(img_data, "save"):
        final_name = img_name if Path(img_name).suffix else f"{img_name}.png"
        ext = Path(final_name).suffix.lower()
        fmt_map = {
            ".png": "PNG",
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".webp": "WEBP",
        }
        fmt = fmt_map.get(ext, "PNG")
        if ext not in fmt_map:
            final_name = f"{Path(final_name).stem}.png"

        buff = io.BytesIO()
        img_data.save(buff, format=fmt)
        return final_name, buff.getvalue()

    raise TypeError(f"Tipo de imagen no soportado: {type(img_data)}")


def build_artifacts(device: str):
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    return create_model_dict(device=device, dtype=model_dtype, attention_implementation="sdpa")


def print_runtime_diagnostics():
    torch_cuda = torch.version.cuda or "None"
    cuda_available = torch.cuda.is_available()

    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch_cuda}")
    print(f"CUDA disponible en PyTorch: {cuda_available}")

    if not cuda_available and torch.version.cuda is None:
        print(
            "Pista: este intérprete tiene una build CPU-only de PyTorch. "
            "Instala una wheel CUDA o ejecuta el script con un Python que ya tenga torch+cu*."
        )
    elif not cuda_available:
        print(
            "Pista: PyTorch incluye runtime CUDA, pero no puede inicializar la GPU en este intérprete. "
            "Revisa driver, variables de entorno y conflictos entre entornos Python."
        )


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def can_attempt_model_reload(min_free_mb: int = 256) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes >= (min_free_mb * 1024 * 1024)
    except Exception:
        return False


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def convert_pdf_in_cuda_chunks(converter_artifacts: dict, pdf_path: Path, initial_chunk_pages: int = 3):
    page_count = get_pdf_page_count(pdf_path)
    if page_count <= 0:
        return "", {}, []

    markdown_parts = []
    images_merged = {}
    cpu_fallback_pages = []

    page_start = 0
    chunk_pages = max(1, initial_chunk_pages)
    max_chunk_cap = max(1, initial_chunk_pages)

    quality_profiles = [
        (96, 160),
        (88, 144),
    ]
    quality_idx = 0
    emergency_retried_pages = set()
    uncapped_cuda_pages = set()
    cpu_artifacts = None

    while page_start < page_count:
        page_end = min(page_start + chunk_pages, page_count)
        page_range = list(range(page_start, page_end))
        lowres_dpi, highres_dpi = quality_profiles[quality_idx]

        config = {
            "page_range": page_range,
            "lowres_image_dpi": lowres_dpi,
            "highres_image_dpi": highres_dpi,
            "layout_batch_size": 1,
            "recognition_batch_size": 3,
            "detection_batch_size": 1,
            "ocr_error_batch_size": 3,
            "ocr_task_name": "ocr_with_boxes",
            "disable_ocr_math": False,
            "min_document_ocr_threshold": 0.85,
        }

        converter = PdfConverter(artifact_dict=converter_artifacts, config=config)

        try:
            rendered = converter(str(pdf_path))
            output = text_from_rendered(rendered)

            if isinstance(output, tuple) and len(output) >= 3:
                chunk_text = output[0]
                chunk_images = output[2] or {}
            else:
                chunk_text = str(output)
                chunk_images = {}

            if isinstance(chunk_images, dict) and chunk_images:
                renamed_images = {}
                for img_name, img_data in chunk_images.items():
                    new_name = f"p{page_start + 1}_{img_name}"
                    final_name, img_bytes = to_image_bytes(new_name, img_data)
                    renamed_images[final_name] = img_bytes
                    chunk_text = chunk_text.replace(img_name, final_name)
                images_merged.update(renamed_images)

            markdown_parts.append(chunk_text)
            print(
                f"  Chunk páginas {page_start + 1}-{page_end}/{page_count}: OK "
                f"(chunk={chunk_pages}, dpi={lowres_dpi}/{highres_dpi})"
            )

            page_start = page_end
            quality_idx = 0

            if chunk_pages < max_chunk_cap:
                chunk_pages = min(max_chunk_cap, chunk_pages + 1)

            if torch.cuda.is_available():
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                print(f"    CUDA memoria -> reservada: {reserved_gb:.2f} GB, asignada: {allocated_gb:.2f} GB")

            cleanup_cuda()

        except Exception as e:
            err = str(e).lower()
            is_oom = "out of memory" in err or ("cuda" in err and "memory" in err)
            if not is_oom:
                raise

            cleanup_cuda()

            if chunk_pages > 1:
                chunk_pages = max(1, chunk_pages // 2)
                max_chunk_cap = min(max_chunk_cap, chunk_pages)
                print(f"  OOM en chunk páginas {page_start + 1}-{page_end}. Reintentando con chunk={chunk_pages}...")
                continue

            if quality_idx < len(quality_profiles) - 1:
                quality_idx += 1
                next_low, next_high = quality_profiles[quality_idx]
                print(
                    "  OOM en chunk=1. Reintentando misma página con "
                    f"dpi={next_low}/{next_high} para mantener CUDA bajo 4GB..."
                )
                continue

            failed_page = page_start + 1

            if page_start not in uncapped_cuda_pages:
                uncapped_cuda_pages.add(page_start)
                try:
                    torch.cuda.set_per_process_memory_fraction(1.0, 0)
                    print(
                        f"  OOM persistente en página {failed_page}. Último recurso CUDA: reintento sin límite de fracción VRAM..."
                    )
                except Exception:
                    pass
                quality_idx = len(quality_profiles) - 1
                chunk_pages = 1
                continue

            if page_start not in emergency_retried_pages and can_attempt_model_reload():
                emergency_retried_pages.add(page_start)
                print(
                    f"  OOM persistente en página {failed_page}. Último recurso: recargar modelos CUDA y reintentar página..."
                )
                cleanup_cuda()
                try:
                    converter_artifacts = build_artifacts("cuda")
                except Exception as reload_error:
                    if "out of memory" in str(reload_error).lower():
                        print("  No hay VRAM suficiente para recargar modelos. Se omite la página.")
                    else:
                        print(f"  Falló recarga de modelos: {reload_error}. Se omite la página.")
                    cleanup_cuda()
                    markdown_parts.append(f"\n\n> [Página {failed_page} omitida por OOM en CUDA]\n")
                    cpu_fallback_pages.append(failed_page)
                    page_start += 1
                    quality_idx = 0
                    chunk_pages = 1
                    max_chunk_cap = 1
                    continue
                quality_idx = len(quality_profiles) - 1
                chunk_pages = 1
                continue

            print(f"  OOM persistente en página {failed_page}. Fallback final: procesar esta página en CPU para no omitirla...")
            try:
                if cpu_artifacts is None:
                    cpu_artifacts = build_artifacts("cpu")

                cpu_config = {
                    "page_range": [page_start],
                    "lowres_image_dpi": 96,
                    "highres_image_dpi": 160,
                    "layout_batch_size": 1,
                    "recognition_batch_size": 1,
                    "detection_batch_size": 1,
                    "ocr_error_batch_size": 1,
                    "ocr_task_name": "ocr_with_boxes",
                    "disable_ocr_math": False,
                    "min_document_ocr_threshold": 0.85,
                }
                cpu_converter = PdfConverter(artifact_dict=cpu_artifacts, config=cpu_config)
                cpu_rendered = cpu_converter(str(pdf_path))
                cpu_output = text_from_rendered(cpu_rendered)

                if isinstance(cpu_output, tuple) and len(cpu_output) >= 3:
                    cpu_text = cpu_output[0]
                    cpu_images = cpu_output[2] or {}
                else:
                    cpu_text = str(cpu_output)
                    cpu_images = {}

                if isinstance(cpu_images, dict) and cpu_images:
                    renamed_images = {}
                    for img_name, img_data in cpu_images.items():
                        new_name = f"p{page_start + 1}_{img_name}"
                        final_name, img_bytes = to_image_bytes(new_name, img_data)
                        renamed_images[final_name] = img_bytes
                        cpu_text = cpu_text.replace(img_name, final_name)
                    images_merged.update(renamed_images)

                markdown_parts.append(cpu_text)
                cpu_fallback_pages.append(failed_page)
                print(f"  Página {failed_page} procesada en CPU (fallback).")
            except Exception as cpu_error:
                print(f"  Fallback CPU falló en página {failed_page}: {cpu_error}")
                markdown_parts.append(f"\n\n> [Página {failed_page} no pudo procesarse en CUDA ni CPU]\n")

            page_start += 1
            quality_idx = 0
            chunk_pages = 1
            max_chunk_cap = 1
            cleanup_cuda()
            try:
                torch.cuda.set_per_process_memory_fraction(0.90, 0)
            except Exception:
                pass

    return "\n\n".join(markdown_parts), images_merged, cpu_fallback_pages


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("--- INICIANDO CONVERSIÓN ---")
    print_runtime_diagnostics()
    print(f"Hardware: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    if device == "cuda":
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if device != "cuda":
        print("No se detecta CUDA. Este modo exige GPU para cumplir el requisito de VRAM/CUDA.")
        return

    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    print("Límite VRAM proceso: 90% (protección para GPU de 4GB)")

    base_dir = Path(__file__).resolve().parent
    docs_dir = base_dir / "docs"
    pdf_dir = docs_dir / "pdf"
    md_dir = docs_dir / "markdown"

    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(docs_dir.glob("*.pdf"))
    move_needed = True
    if not pdfs:
        pdfs = list(pdf_dir.glob("*.pdf"))
        move_needed = False

    if not pdfs:
        print("No se han encontrado PDFs en /docs o /docs/pdf")
        return

    print("Cargando modelos en memoria (modo cuda, fp16, attention=sdpa)...")
    converter_artifacts = build_artifacts("cuda")

    for pdf_path in sorted(pdfs):
        dest_pdf = pdf_dir / pdf_path.name
        if move_needed:
            print(f"\nMoviendo {pdf_path.name} a docs/pdf/...")
            try:
                shutil.move(str(pdf_path), str(dest_pdf))
            except Exception:
                pass
        else:
            dest_pdf = pdf_path

        safe_stem = sanitize_stem(dest_pdf.stem)
        out_dir = md_dir / safe_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        md_file = out_dir / f"{safe_stem}.md"

        if md_file.exists() and md_file.stat().st_size > 500:
            print(f"Saltando {safe_stem} (ya existe).")
            continue

        print(f"Procesando: {dest_pdf.name}...")

        try:
            cleanup_cuda()
            full_text, images_dict, cpu_fallback_pages = convert_pdf_in_cuda_chunks(
                converter_artifacts,
                dest_pdf,
                initial_chunk_pages=3,
            )

            md_file.write_text(full_text, encoding="utf-8")

            img_count = 0
            if isinstance(images_dict, dict) and images_dict:
                for img_name, img_data in images_dict.items():
                    (out_dir / img_name).write_bytes(img_data)
                    img_count += 1

            if cpu_fallback_pages:
                print(f"  Finalizado: {md_file.name} (+{img_count} imágenes, páginas en fallback CPU: {cpu_fallback_pages})")
            else:
                print(f"  Finalizado: {md_file.name} (+{img_count} imágenes)")

            reserved_gb = torch.cuda.memory_reserved() / 1e9
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  CUDA memoria -> reservada: {reserved_gb:.2f} GB, asignada: {allocated_gb:.2f} GB")

        except Exception as e:
            print(f"  ERROR en {dest_pdf.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("  Continuando con el siguiente PDF sin recargar modelos (evita crash por OOM en recarga).")
            cleanup_cuda()
        finally:
            cleanup_cuda()

    print("\n--- PROCESO COMPLETADO ---")


if __name__ == "__main__":
    main()
