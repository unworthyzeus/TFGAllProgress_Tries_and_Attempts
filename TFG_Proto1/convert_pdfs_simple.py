import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MODEL_DTYPE"] = "fp16"
os.environ["LAYOUT_BATCH_SIZE"] = "1"
os.environ["RECOGNITION_BATCH_SIZE"] = "1"
os.environ["DETECTOR_BATCH_SIZE"] = "1"
os.environ["ORDER_BATCH_SIZE"] = "1"

import shutil
from pathlib import Path
import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- MARKER conversion (v1.x) ---")
    print(f"Usando: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    DOCS_DIR = Path("docs")
    PDF_DIR = DOCS_DIR / "pdf"
    MD_DIR = DOCS_DIR / "markdown"
    
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs: pdfs = list(PDF_DIR.glob("*.pdf"))

    print(f"Convertiendo {len(pdfs)} PDFs. Cargando modelos...")
    model_dict = create_model_dict()
    # Forzar que NO use procesos paralelos para que no pete la VRAM
    converter = PdfConverter(artifact_dict=model_dict)

    for pdf_path in sorted(pdfs):
        print(f"\n>> {pdf_path.name}")
        out_root = MD_DIR / pdf_path.stem
        out_root.mkdir(parents=True, exist_ok=True)
        md_file = out_root / f"{pdf_path.stem}.md"

        if md_file.exists() and md_file.stat().st_size > 100:
            print("   Ya existe, saltando.")
            continue

        try:
            rendered = converter(str(pdf_path))
            # En Marker 1.x, rendered tiene .markdown y .images
            # Pero text_from_rendered es el oficial
            full_text, images, metadata = text_from_rendered(rendered)

            md_file.write_text(full_text, encoding="utf-8")
            if isinstance(images, dict):
                for img_name, img_data in images.items():
                    (out_root / img_name).write_bytes(img_data)
                print(f"   Hecho: {len(images)} imágenes.")
            else:
                print("   Hecho (sin imágenes).")

        except Exception as e:
            print(f"   ERROR: {str(e)}")
        finally:
            if device == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
