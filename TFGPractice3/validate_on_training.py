"""
Validate trained CKM model on a TRAINING city (Beijing) to check for overfitting vs generalization.

If results are good here but bad on SF, it confirms domain gap issue.
If results are bad here too, there's a training/model problem.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "ckm_epoch_10.pth"  # Change to test different checkpoints

MANIFEST_FILE = r"c:/TFG/TFGPractice/CKMImagenet/Datasets/train_manifest.csv"
DATA_ROOT = r"c:/TFG/TFGPractice/CKMImagenet"

# Filter to validate on specific city
VALIDATE_CITY = "Beijing"  # Change to "Boston", "College", etc.

BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 500  # Use subset for quick validation

TARGET_COLS = ["ch_gain", "AOA_phi", "AOA_theta", "AOD_phi", "AOD_theta"]

# ==========================================
# MODEL DEFINITION (Must match training!)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, 1)

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

# ==========================================
# DATASET
# ==========================================
class CityValidationDataset(Dataset):
    def __init__(self, manifest_csv, root_dir, city_filter, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(manifest_csv)
        
        # Filter to specific city
        self.df = self.df[self.df['input_path'].str.contains(city_filter, case=False)]
        
        # Filter to only samples with ALL targets present
        valid_mask = self.df[TARGET_COLS].notna().all(axis=1)
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        # Limit samples
        if max_samples and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} samples for {city_filter}")

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

# Standard Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ==========================================
# METRICS
# ==========================================
def compute_psnr(mse, max_val=1.0):
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10((max_val ** 2) / mse)

# ==========================================
# MAIN
# ==========================================
def main():
    # Load model
    model = UNet(n_channels=1, n_classes=len(TARGET_COLS)).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}. Running on {DEVICE}")
    
    # Load dataset
    dataset = CityValidationDataset(
        MANIFEST_FILE, DATA_ROOT, VALIDATE_CITY, 
        transform=transform, max_samples=NUM_SAMPLES
    )
    
    if len(dataset) == 0:
        print(f"No valid samples found for {VALIDATE_CITY}")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Accumulators
    channel_mse = {col: [] for col in TARGET_COLS}
    channel_mae = {col: [] for col in TARGET_COLS}
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            outputs = model(inputs)
            
            # Per-channel metrics
            for i, col in enumerate(TARGET_COLS):
                pred = outputs[:, i, :, :]
                gt = targets[:, i, :, :]
                
                mse = ((pred - gt) ** 2).mean(dim=(1, 2))
                mae = (pred - gt).abs().mean(dim=(1, 2))
                
                channel_mse[col].extend(mse.cpu().numpy())
                channel_mae[col].extend(mae.cpu().numpy())
    
    # Print results
    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS ON {VALIDATE_CITY.upper()} (TRAINING DATA)")
    print("=" * 60)
    print(f"{'Channel':<20} {'MSE':>12} {'MAE':>12} {'PSNR (dB)':>12}")
    print("-" * 60)
    
    all_mse = []
    for col in TARGET_COLS:
        mse = np.mean(channel_mse[col])
        mae = np.mean(channel_mae[col])
        psnr = compute_psnr(mse)
        all_mse.append(mse)
        print(f"{col:<20} {mse:>12.6f} {mae:>12.6f} {psnr:>12.2f}")
    
    print("-" * 60)
    avg_mse = np.mean(all_mse)
    avg_psnr = compute_psnr(avg_mse)
    print(f"{'AVERAGE':<20} {avg_mse:>12.6f} {'-':>12} {avg_psnr:>12.2f}")
    print("=" * 60)
    print(f"\nTotal samples evaluated: {len(dataset)}")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    if avg_mse < 0.01:
        print("✓ GOOD: Model works well on training data.")
        print("  → Poor SF results indicate DOMAIN GAP (expected).")
    elif avg_mse < 0.05:
        print("~ OKAY: Model partially learned training data.")
        print("  → May need more epochs or tuning.")
    else:
        print("✗ BAD: Model doesn't work well even on training data.")
        print("  → Check for bugs in data loading or model architecture.")
    print("-" * 60)

if __name__ == "__main__":
    main()
