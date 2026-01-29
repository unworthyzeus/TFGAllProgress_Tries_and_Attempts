import os
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = "Datasets"
OUTPUT_CSV = "Datasets/ckm_dataset_manifest.csv"

# Grid size for converting row/col to graph_id
GRID_COLS = 9  # Based on "81 partial overlapping subareas" (9x9)

# Mapping logic:
# Environment files look like: BJ_128_BS1_0_0.png (City_..._BS{id}_{row}_{col}.png)
# Target files look like: graph_0.png (graph_{id}.png)
# We assume id = row * 9 + col (Standard row-major order)
# ==========================================    
PARAM_KEYWORDS: Dict[str, List[str]] = {
    "ch_gain": ["ch_gain", "channel_gain"],
    "AOA_phi": ["aoa_phi", "aoa"],
    "AOA_theta": ["aoa_theta"],
    "AOD_phi": ["aod_phi", "aod"],
    "AOD_theta": ["aod_theta"],
}


def parse_env_filename(filename: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract BS, row, col counters from environment tile names."""
    match = re.search(r"BS(\d+)_(\d+)_(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def discover_env_dirs(root: str) -> List[str]:
    """Return every directory that looks like an environment image folder."""
    env_dirs: List[str] = []
    for current_root, dirs, _ in os.walk(root):
        for directory in dirs:
            name = directory.lower()
            if "image" not in name:
                continue
            # Exclude parameter folders that might have "image" in their name (e.g. Paris_3_image_128_AoA)
            if any(term in name for term in ("aoa", "aod", "gain")):
                continue
            env_dirs.append(os.path.join(current_root, directory))
    return env_dirs


def find_param_folder(parent_dir: str, env_name: str, keywords: List[str], strict: bool = False) -> Optional[str]:
    """
    Find a parameter folder in parent_dir matching keywords.
    Prioritizes folders related to env_name (Specific Match).
    If strict is True, disables General Match (fallback).
    """
    candidates = []
    try:
        entries = os.listdir(parent_dir)
    except FileNotFoundError:
        return None

    for entry in entries:
        entry_path = os.path.join(parent_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        
        entry_lower = entry.lower()
        if any(keyword in entry_lower for keyword in keywords):
            candidates.append(entry)

    if not candidates:
        return None

    # 1. Specific Match: Starts with env_name (e.g. Paris_3_image_128_AoA starts with Paris_3_image_128)
    for cand in candidates:
        if cand.startswith(env_name):
            return cand

    # 2. Prefix Match: Starts with env_name prefix (e.g. London1_ch_gain starts with London1_)
    # Extract prefix from env_name (remove 'image_128' or similar)
    prefix_match = re.match(r"(.+?)(_image|_img)", env_name, re.IGNORECASE)
    if prefix_match:
        prefix = prefix_match.group(1)
        for cand in candidates:
            if cand.startswith(prefix):
                return cand

    # 3. General Match: Return the first candidate (Fallback for Beijing)
    if not strict:
        return candidates[0]
    
    return None


def find_bs_subfolder(param_root: str, bs_id: int) -> Optional[str]:
    # This function is deprecated by the pre-scanning logic in main()
    pass

def select_graph_filename(bs_folder: str, row: int, col: int) -> Optional[str]:
    """Choose the first graph filename that exists for the provided row/col."""
    candidates = [
        f"graph_{row * GRID_COLS + col}.png",
        f"graph_{row * GRID_COLS + col + 1}.png",
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(bs_folder, candidate)):
            return candidate
    return None

def find_target_file(param_dir: str, env_filename: str, bs_id: int, row: int, col: int, param_name: str, bs_map: Dict[int, str]) -> Optional[str]:
    """
    Find the target file using Strategy A (BS folders) or Strategy B (Flat files).
    """
    # Strategy A: Check if we have a mapped BS folder
    if bs_map:
        bs_subfolder = bs_map.get(bs_id)
        if bs_subfolder:
            bs_path = os.path.join(param_dir, bs_subfolder)
            filename = select_graph_filename(bs_path, row, col)
            if filename:
                return os.path.join(bs_subfolder, filename)
        return None
    else:
        # Strategy B: Flat structure (Paris/Nanjing/Miami)
        # Construct filename: {EnvStem}_{Suffix}.png
        
        # Determine suffix from param_name
        suffix = ""
        param_lower = param_name.lower()
        if "aoa" in param_lower:
            suffix = "_AoA"
        elif "aod" in param_lower:
            suffix = "_AoD"
        elif "ch_gain" in param_lower:
            suffix = "_ch_gain"
        
        env_stem = os.path.splitext(env_filename)[0]
        candidate = f"{env_stem}{suffix}.png"
        
        if os.path.exists(os.path.join(param_dir, candidate)):
            return candidate
            
    return None


def scan_bs_folders(param_dir: str) -> Dict[int, str]:
    """
    Scan a parameter directory and return a map of BS ID -> Folder Name.
    Returns empty dict if no BS folders found (implies Flat structure).
    """
    bs_map = {}
    try:
        entries = os.listdir(param_dir)
    except FileNotFoundError:
        return {}

    for entry in entries:
        entry_path = os.path.join(param_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        
        # Check for BS pattern (e.g. Beijing_ch_gain_BS10 or just BS10)
        # We look for 'BS' followed by digits at the end or somewhere
        match = re.search(r"BS(\d+)", entry, re.IGNORECASE)
        if match:
            bs_id = int(match.group(1))
            bs_map[bs_id] = entry
            
    return bs_map


def main() -> None:
    records: List[Dict[str, object]] = []

    print(f"Scanning for data in {os.path.abspath(DATA_ROOT)}...")

    env_dirs = discover_env_dirs(DATA_ROOT)
    if not env_dirs:
        print("No environment directories found. Double-check DATA_ROOT.")
        return

    # Pre-calculate env counts per city to determine strictness
    city_env_counts = {}
    for env_dir in env_dirs:
        city_root = os.path.dirname(env_dir)
        city_env_counts[city_root] = city_env_counts.get(city_root, 0) + 1

    for env_dir in env_dirs:
        city_root = os.path.dirname(env_dir)
        env_folder_name = os.path.basename(env_dir)
        
        print(f"Processing environment: {env_folder_name} in {city_root}")

        is_multi_env = city_env_counts.get(city_root, 0) > 1

        # Identify parameter folders for this environment
        param_folders = {}
        param_bs_maps = {} # Cache BS maps for each param
        
        for param_name, keywords in PARAM_KEYWORDS.items():
            folder_name = find_param_folder(city_root, env_folder_name, keywords, strict=is_multi_env)
            
            # Fallback for Theta if not found
            if not folder_name and param_name in ["AOA_theta", "AOD_theta"]:
                fallback_keywords = ["aoa"] if param_name == "AOA_theta" else ["aod"]
                folder_name = find_param_folder(city_root, env_folder_name, fallback_keywords, strict=is_multi_env)
                
            param_folders[param_name] = folder_name
            
            # Pre-scan for BS folders if param folder exists
            if folder_name:
                param_dir = os.path.join(city_root, folder_name)
                param_bs_maps[param_name] = scan_bs_folders(param_dir)
            else:
                param_bs_maps[param_name] = {}

        # Scan images in environment folder
        try:
            env_files = os.listdir(env_dir)
        except FileNotFoundError:
            continue

        for filename in env_files:
            if not filename.lower().endswith(".png"):
                continue

            bs_id, row, col = parse_env_filename(filename)
            if bs_id is None:
                continue

            # Build record
            record = {
                "input_path": os.path.join(env_dir, filename),
                "bs_id": bs_id,
                "row": row,
                "col": col
            }

            # Find targets
            for param_name in PARAM_KEYWORDS.keys():
                folder_name = param_folders.get(param_name)
                target_rel_path = None
                
                if folder_name:
                    param_dir = os.path.join(city_root, folder_name)
                    bs_map = param_bs_maps.get(param_name, {})
                    target_file = find_target_file(param_dir, filename, bs_id, row, col, param_name, bs_map)
                    if target_file:
                        target_rel_path = os.path.join(city_root, folder_name, target_file)
                
                record[param_name] = target_rel_path
                
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Filter out completely broken records? 
    # Or just save.
    
    print(f"Found {len(df)} samples.")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Warning: Some samples have missing targets:")
        print(missing_counts[missing_counts > 0])
        print("Note: Training script may fail on these samples unless modified.")

    # Fill NaNs with empty string to avoid CSV issues, though training script needs handling
    df.fillna("", inplace=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Manifest saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
