"""
Tile the San Francisco city map image into 128x128 sub-images.

The naming convention follows the pattern used in other cities:
    SF_128_BS{bs_id}_{row}_{col}.png

We use a 9x9 grid (81 tiles, indices 1-81).
The graph_id = row * 9 + col + 1 (1-indexed to match graph_{id}.png).

This script scans the existing target folders (SF_ch_gain, SF_AOA_phi, etc.)
to determine which BS IDs and which graph IDs actually exist, then creates
tiles only for those combinations.
"""

import os
from PIL import Image
import re


# ==========================================
# CONFIGURATION
# ==========================================
SF_ROOT = "./SF"
INPUT_IMAGE = os.path.join(SF_ROOT, "San Fransisco.jpg")
OUTPUT_DIR = os.path.join(SF_ROOT, "SF_image_128")

TILE_SIZE = 128
GRID_SIZE = 9  # 9x9 grid

# ==========================================
# DISCOVER EXISTING TARGET DATA
# ==========================================
def discover_existing_targets(sf_root: str):
    """
    Scan target folders to find which (bs_id, graph_id) pairs exist.
    Returns a set of (bs_id, graph_id) tuples.
    """
    existing = set()
    
    # Look at any parameter folder (they should all have the same structure)
    param_folders = ["SF_ch_gain", "SF_AOA_phi", "SF_AOA_theta", "SF_AOD_phi", "SF_AOD_theta"]
    
    for param_folder in param_folders:
        param_path = os.path.join(sf_root, param_folder)
        if not os.path.isdir(param_path):
            continue
        
        # List BS subfolders
        for bs_folder in os.listdir(param_path):
            bs_path = os.path.join(param_path, bs_folder)
            if not os.path.isdir(bs_path):
                continue
            
            # Extract BS ID from folder name
            match = re.search(r"BS(\d+)", bs_folder)
            if not match:
                continue
            bs_id = int(match.group(1))
            
            # List graph files
            for filename in os.listdir(bs_path):
                graph_match = re.match(r"graph_(\d+)\.png", filename)
                if graph_match:
                    graph_id = int(graph_match.group(1))
                    existing.add((bs_id, graph_id))
    
    return existing


def graph_id_to_row_col(graph_id: int, grid_size: int = 9):
    """Convert 1-indexed graph_id to (row, col) indices (0-indexed)."""
    # graph_1 -> row=0, col=0
    # graph_9 -> row=0, col=8
    # graph_10 -> row=1, col=0
    idx = graph_id - 1  # Convert to 0-indexed
    row = idx // grid_size
    col = idx % grid_size
    return row, col


def tile_image(img: Image.Image, row: int, col: int, tile_size: int = 128, grid_size: int = 9) -> Image.Image:
    """
    Extract a tile from the image.
    
    The image is divided into grid_size x grid_size logical cells.
    Each cell is scaled to tile_size x tile_size.
    """
    img_width, img_height = img.size
    
    # Calculate cell size in original image
    cell_width = img_width / grid_size
    cell_height = img_height / grid_size
    
    # Calculate bounding box for this cell
    left = int(col * cell_width)
    upper = int(row * cell_height)
    right = int((col + 1) * cell_width)
    lower = int((row + 1) * cell_height)
    
    # Crop and resize
    tile = img.crop((left, upper, right, lower))
    tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    
    return tile


def main():
    print(f"Loading source image: {INPUT_IMAGE}")
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"ERROR: Source image not found: {INPUT_IMAGE}")
        return
    
    img = Image.open(INPUT_IMAGE)
    print(f"  Image size: {img.size}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Discover existing targets
    print("Scanning existing target data...")
    existing_pairs = discover_existing_targets(SF_ROOT)
    print(f"  Found {len(existing_pairs)} unique (BS, graph_id) pairs")
    
    if not existing_pairs:
        print("WARNING: No existing target data found. Creating all possible tiles...")
        # Create tiles for all possible combinations
        bs_ids = range(1, 22)  # BS1 to BS21
        graph_ids = range(1, 82)  # graph_1 to graph_81
        existing_pairs = {(bs, g) for bs in bs_ids for g in graph_ids}
    
    # Get unique BS IDs and graph IDs
    bs_ids = sorted(set(bs for bs, _ in existing_pairs))
    graph_ids = sorted(set(g for _, g in existing_pairs))
    
    print(f"  BS IDs: {min(bs_ids)} - {max(bs_ids)} ({len(bs_ids)} total)")
    print(f"  Graph IDs: {min(graph_ids)} - {max(graph_ids)} ({len(graph_ids)} total)")
    
    # Generate tiles
    tiles_created = 0
    for bs_id in bs_ids:
        for graph_id in graph_ids:
            if (bs_id, graph_id) not in existing_pairs:
                continue
            
            row, col = graph_id_to_row_col(graph_id, GRID_SIZE)
            
            # Generate tile
            tile = tile_image(img, row, col, TILE_SIZE, GRID_SIZE)
            
            # Save with naming convention: SF_128_BS{bs_id}_{row}_{col}.png
            filename = f"SF_128_BS{bs_id}_{row}_{col}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Convert to grayscale to match other cities
            tile_gray = tile.convert('L')
            tile_gray.save(filepath)
            tiles_created += 1
    
    print(f"\nDone! Created {tiles_created} tiles in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
