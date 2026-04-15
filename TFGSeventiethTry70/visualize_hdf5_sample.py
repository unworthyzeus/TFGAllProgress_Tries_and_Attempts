from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIELDS: Tuple[str, ...] = (
    "topology_map",
    "path_loss",
    "delay_spread",
    "angular_spread",
    "los_mask",
)


def list_sample_refs(hdf5_path: Path) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []
    with h5py.File(hdf5_path, "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))
    return refs


def load_sample(hdf5_path: Path, city: str, sample: str) -> Dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        group = handle[city][sample]
        return {field: np.asarray(group[field][...]) for field in FIELDS}


def to_preview_image(field_name: str, array: np.ndarray, preview_size: int) -> Image.Image:
    arr = np.asarray(array, dtype=np.float32)
    if field_name == "los_mask":
        preview = (arr > 0.5).astype(np.uint8) * 255
    else:
        min_value = float(np.min(arr))
        max_value = float(np.max(arr))
        if max_value - min_value < 1e-12:
            preview = np.zeros_like(arr, dtype=np.uint8)
        else:
            preview = ((arr - min_value) / (max_value - min_value) * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(preview, mode="L").convert("RGB")
    return image.resize((preview_size, preview_size), Image.Resampling.NEAREST)


def add_label(image: Image.Image, title: str) -> Image.Image:
    label_height = 28
    canvas = Image.new("RGB", (image.width, image.height + label_height), (255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, label_height), fill=(240, 240, 240))
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    return canvas


def build_contact_sheet(sample_data: Dict[str, np.ndarray], preview_size: int) -> Image.Image:
    labeled_tiles = [add_label(to_preview_image(field, sample_data[field], preview_size), field) for field in FIELDS]
    columns = 3
    rows = (len(labeled_tiles) + columns - 1) // columns
    tile_width = labeled_tiles[0].width
    tile_height = labeled_tiles[0].height
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (225, 225, 225))

    for index, tile in enumerate(labeled_tiles):
        row = index // columns
        col = index % columns
        sheet.paste(tile, (col * tile_width, row * tile_height))
    return sheet


def print_stats(city: str, sample: str, sample_data: Dict[str, np.ndarray]) -> None:
    print(f"Sample: {city}/{sample}")
    for field in FIELDS:
        arr = np.asarray(sample_data[field])
        print(
            f"  - {field}: shape={arr.shape}, dtype={arr.dtype}, min={float(arr.min()):.4f}, "
            f"max={float(arr.max()):.4f}, mean={float(arr.mean()):.4f}"
        )


def sanitize_fragment(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_").replace("\\", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one sample from CKM_Dataset.h5.")
    parser.add_argument("--hdf5", default="CKM_Dataset.h5", help="Path to the HDF5 dataset.")
    parser.add_argument("--city", help="City name to visualize.")
    parser.add_argument("--sample", help="Sample name inside the city, for example sample_00001.")
    parser.add_argument("--index", type=int, default=0, help="Global sample index if city/sample are not provided.")
    parser.add_argument("--preview-size", type=int, default=320, help="Tile size for each map preview.")
    parser.add_argument("--output", default="outputs/dataset_preview", help="Directory where the PNG preview will be written.")
    parser.add_argument("--list-cities", action="store_true", help="List the available city groups and exit.")
    parser.add_argument("--list-samples", help="List sample names for one city and exit.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of samples to print with --list-samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        if args.list_cities:
            for city in sorted(handle.keys()):
                print(city)
            return

        if args.list_samples:
            if args.list_samples not in handle:
                raise KeyError(f"City not found: {args.list_samples}")
            for sample_name in list(sorted(handle[args.list_samples].keys()))[: max(args.limit, 1)]:
                print(sample_name)
            return

    if args.city and args.sample:
        city, sample = args.city, args.sample
    else:
        refs = list_sample_refs(hdf5_path)
        if not refs:
            raise RuntimeError("No samples were found in the HDF5 dataset.")
        index = max(0, min(args.index, len(refs) - 1))
        city, sample = refs[index]

    sample_data = load_sample(hdf5_path, city, sample)
    print_stats(city, sample, sample_data)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{sanitize_fragment(city)}_{sanitize_fragment(sample)}_preview.png"
    output_path = output_dir / output_name
    build_contact_sheet(sample_data, preview_size=max(args.preview_size, 64)).save(output_path)
    print(f"Saved preview to: {output_path}")


if __name__ == "__main__":
    main()