
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random

# Configuration
MANIFEST_FILE = r"c:/TFG/TFGPractice/CKMImagenet/Datasets/ckm_dataset_manifest.csv"
DATA_ROOT = r"c:/TFG/TFGPractice/CKMImagenet"
TARGET_COLS = ["ch_gain"]
USE_AUGMENTATION = False
AUG_HFLIP_PROB = 0.5
AUG_VFLIP_PROB = 0.5
AUG_ROTATE_PROB = 0.5

class CKMManifestDataset(Dataset):
    def __init__(self, manifest_csv, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        
        self.df = pd.read_csv(manifest_csv)
        print(f"Loaded Dataset: {len(self.df)} samples from {manifest_csv}")

    def __len__(self):
        return len(self.df)
    
    def _apply_augmentation(self, images):
        if random.random() < AUG_HFLIP_PROB:
            images = [TF.hflip(img) for img in images]
        if random.random() < AUG_VFLIP_PROB:
            images = [TF.vflip(img) for img in images]
        if random.random() < AUG_ROTATE_PROB:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        return images

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        input_path = os.path.join(self.root_dir, row['input_path'])
        input_img = Image.open(input_path).convert('L')
        input_img = input_img.resize((128, 128), Image.BILINEAR)
        
        target_images = []
        mask_tensors = []

        for col in TARGET_COLS:
            path_val = row[col]
            if pd.isna(path_val):
                target_images.append(None)
                mask_tensors.append(torch.zeros((1, 128, 128)))
            else:
                t_path = os.path.join(self.root_dir, path_val)
                try:
                    img = Image.open(t_path).convert('L')
                    img = img.resize((128, 128), Image.BILINEAR)
                    target_images.append(img)
                    mask_tensors.append(torch.ones((1, 128, 128)))
                except (FileNotFoundError, OSError):
                    target_images.append(None)
                    mask_tensors.append(torch.zeros((1, 128, 128)))

        if self.augment:
            all_images = [input_img]
            valid_indices = []
            for i, img in enumerate(target_images):
                if img is not None:
                    all_images.append(img)
                    valid_indices.append(i)
            
            all_images = self._apply_augmentation(all_images)
            
            input_img = all_images[0]
            for j, idx_t in enumerate(valid_indices):
                target_images[idx_t] = all_images[j + 1]

        input_tensor = TF.to_tensor(input_img)
        
        target_tensors = []
        for i, img in enumerate(target_images):
            if img is not None:
                target_tensors.append(TF.to_tensor(img))
            else:
                target_tensors.append(torch.zeros((1, 128, 128)))

        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.cat(mask_tensors, dim=0)
        
        return input_tensor, target_tensor, mask_tensor

def main():
    print("Testing Dataset Setup for TFGPractice2 (Power Only)...")
    try:
        dataset = CKMManifestDataset(MANIFEST_FILE, DATA_ROOT, augment=USE_AUGMENTATION)
        print("Dataset initialized successfully.")
        
        if len(dataset) > 0:
            sample_idx = 0
            input_t, target_t, mask_t = dataset[sample_idx]
            print(f"Sample {sample_idx} loaded.")
            print(f"Input shape: {input_t.shape}")
            print(f"Target shape: {target_t.shape}") # Expect [1, 128, 128]
            print(f"Mask shape: {mask_t.shape}")     # Expect [1, 128, 128]
            
            if target_t.shape[0] == 1:
                print("SUCCESS: Target has 1 channel (Power/Ch_Gain only).")
            else:
                print(f"FAILURE: Target has {target_t.shape[0]} channels (Expected 1).")
        else:
            print("Dataset is empty.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
