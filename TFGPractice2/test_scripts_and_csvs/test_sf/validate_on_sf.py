"""
Validate trained CKM model on San Francisco test set.

This script loads the trained model and evaluates it on the SF manifest,
computing MSE, MAE, and PSNR metrics for each output channel.
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
# Path to trained model (adjust as needed)
MODEL_PATH = "../../ckm_epoch_with_augmentation_2.pth"  # Correct relative path

# SF test manifest
MANIFEST_FILE = "ckm_dataset_manifest.csv"
DATA_ROOT = r"c:/TFG/TFGPractice/test"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COLS = ["ch_gain"]

# ==========================================
# MODEL DEFINITION (Must match training!)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)
        
    def forward(self, x): return self.double_conv(x)

class UNetLarge(nn.Module):
    """
    Larger U-Net with more parameters for better generalization.
    Base channels: 64 -> 96 (1.5x more parameters in first layers)
    Added dropout for regularization
    ~31M parameters (vs ~17M original)
    """
    def __init__(self, n_channels, n_classes, base_channels=96):
        super(UNetLarge, self).__init__()
        bc = base_channels  # 96
        
        # Encoder (more channels)
        self.inc = DoubleConv(n_channels, bc)                    # 1 -> 96
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc, bc*2))      # 96 -> 192
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc*2, bc*4))    # 192 -> 384
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc*4, bc*8, dropout=0.1))   # 384 -> 768
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc*8, bc*16, dropout=0.2))  # 768 -> 1536
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(bc*16, bc*8, 2, stride=2)
        self.conv1 = DoubleConv(bc*16, bc*8, dropout=0.1)        # 1536 -> 768
        self.up2 = nn.ConvTranspose2d(bc*8, bc*4, 2, stride=2)
        self.conv2 = DoubleConv(bc*8, bc*4)                       # 768 -> 384
        self.up3 = nn.ConvTranspose2d(bc*4, bc*2, 2, stride=2)
        self.conv3 = DoubleConv(bc*4, bc*2)                       # 384 -> 192
        self.up4 = nn.ConvTranspose2d(bc*2, bc, 2, stride=2)
        self.conv4 = DoubleConv(bc*2, bc)                         # 192 -> 96
        
        self.outc = nn.Conv2d(bc, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5); x = torch.cat([x4, x], dim=1); x = self.conv1(x)
        x = self.up2(x); x = torch.cat([x3, x], dim=1); x = self.conv2(x)
        x = self.up3(x); x = torch.cat([x2, x], dim=1); x = self.conv3(x)
        x = self.up4(x); x = torch.cat([x1, x], dim=1); x = self.conv4(x)
        return self.outc(x)

# Alias for backward compatibility
UNet = UNetLarge

# ==========================================
# DATASET
# ==========================================
class SFTestDataset(Dataset):
    def __init__(self, manifest_csv, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(manifest_csv)
        
        # Filter to only samples with ALL targets present
        valid_mask = self.df[TARGET_COLS].notna().all(axis=1)
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} valid test samples from {manifest_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Input
        input_path = os.path.join(self.root_dir, row['input_path'])
        input_img = Image.open(input_path).convert('L')
        
        # Targets
        target_tensors = []
        for col in TARGET_COLS:
            t_path = os.path.join(self.root_dir, row[col])
            img = Image.open(t_path).convert('L')
            if self.transform:
                t_tensor = self.transform(img)
            else:
                t_tensor = transforms.ToTensor()(img)
            target_tensors.append(t_tensor)
        
        if self.transform:
            input_tensor = self.transform(input_img)
        else:
            input_tensor = transforms.ToTensor()(input_img)
        
        target_tensor = torch.cat(target_tensors, dim=0)
        
        return input_tensor, target_tensor

# ==========================================
# METRICS
# ==========================================
def compute_psnr(mse, max_val=1.0):
    """Compute PSNR from MSE."""
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)

def validate():
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Available .pth files:")
        for f in os.listdir("../CKMImagenet"):
            if f.endswith(".pth"):
                print(f"  ../CKMImagenet/{f}")
        return
    
    # Load model
    model = UNet(n_channels=1, n_classes=len(TARGET_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print(f"Model loaded. Running on {DEVICE}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    try:
        dataset = SFTestDataset(MANIFEST_FILE, DATA_ROOT, transform=transform)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return
    
    if len(dataset) == 0:
        print("ERROR: No valid samples in test set!")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Metrics accumulators (per channel)
    channel_mse = {col: [] for col in TARGET_COLS}
    channel_mae = {col: [] for col in TARGET_COLS}
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            
            # Compute per-channel metrics
            for i, col in enumerate(TARGET_COLS):
                pred = outputs[:, i, :, :]
                gt = targets[:, i, :, :]
                
                mse = ((pred - gt) ** 2).mean(dim=(1, 2))  # Per-sample MSE
                mae = (pred - gt).abs().mean(dim=(1, 2))   # Per-sample MAE
                
                channel_mse[col].extend(mse.cpu().numpy())
                channel_mae[col].extend(mae.cpu().numpy())
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS ON SAN FRANCISCO TEST SET")
    print("=" * 60)
    print(f"{'Channel':<15} {'MSE':>12} {'MAE':>12} {'PSNR (dB)':>12}")
    print("-" * 60)
    
    overall_mse = []
    for col in TARGET_COLS:
        mse = np.mean(channel_mse[col])
        mae = np.mean(channel_mae[col])
        psnr = compute_psnr(mse)
        overall_mse.append(mse)
        print(f"{col:<15} {mse:>12.6f} {mae:>12.6f} {psnr:>12.2f}")
    
    print("-" * 60)
    avg_mse = np.mean(overall_mse)
    avg_psnr = compute_psnr(avg_mse)
    print(f"{'AVERAGE':<15} {avg_mse:>12.6f} {'-':>12} {avg_psnr:>12.2f}")
    print("=" * 60)
    
    print(f"\nTotal samples evaluated: {len(dataset)}")

if __name__ == "__main__":
    validate()
