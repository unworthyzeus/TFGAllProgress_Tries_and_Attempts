import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
from tqdm import tqdm
import random

# ==========================================
# CONFIGURATION
# ==========================================
MANIFEST_FILE = "Datasets/ckm_dataset_manifest.csv"
DATA_ROOT = "." # Root where the 'Beijing', 'Shanghai' folders are.

BATCH_SIZE = 16          # Reduced for larger model on 4GB VRAM
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set to None to use all data, or a number like 2000 for quick testing
SUBSET_SIZE = 32000

# Data augmentation settings
USE_AUGMENTATION = True
AUG_HFLIP_PROB = 0.5      # Horizontal flip probability
AUG_VFLIP_PROB = 0.5      # Vertical flip probability  
AUG_ROTATE_PROB = 0.5     # Rotation probability (90, 180, 270 degrees)

TARGET_COLS = ["ch_gain", "AOA_phi", "AOA_theta", "AOD_phi", "AOD_theta"]

# ==========================================
# DATASET WITH AUGMENTATION
# ==========================================
class CKMManifestDataset(Dataset):
    def __init__(self, manifest_csv, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        
        # Load Manifest
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}. Run create_manifest.py first!")
        
        self.df = pd.read_csv(manifest_csv)
        print(f"Loaded Dataset: {len(self.df)} samples from {manifest_csv}")

    def __len__(self):
        return len(self.df)
    
    def _apply_augmentation(self, images):
        """Apply the same random augmentation to all images (input + targets)"""
        # Horizontal flip
        if random.random() < AUG_HFLIP_PROB:
            images = [TF.hflip(img) for img in images]
        
        # Vertical flip
        if random.random() < AUG_VFLIP_PROB:
            images = [TF.vflip(img) for img in images]
        
        # Random 90-degree rotation
        if random.random() < AUG_ROTATE_PROB:
            angle = random.choice([90, 180, 270])
            images = [TF.rotate(img, angle) for img in images]
        
        return images

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Input Path
        input_path = os.path.join(self.root_dir, row['input_path'])
        input_img = Image.open(input_path).convert('L')
        
        # Resize input
        input_img = input_img.resize((128, 128), Image.BILINEAR)
        
        # Target Paths & Masks
        target_images = []
        mask_tensors = []

        for col in TARGET_COLS:
            path_val = row[col]
            
            if pd.isna(path_val):
                # Missing target: Create zero tensor and zero mask
                target_images.append(None)
                mask_tensors.append(torch.zeros((1, 128, 128)))
            else:
                # Valid target: Load image
                t_path = os.path.join(self.root_dir, path_val)
                try:
                    img = Image.open(t_path).convert('L')
                    img = img.resize((128, 128), Image.BILINEAR)
                    target_images.append(img)
                    mask_tensors.append(torch.ones((1, 128, 128)))
                except (FileNotFoundError, OSError):
                    target_images.append(None)
                    mask_tensors.append(torch.zeros((1, 128, 128)))

        # Apply augmentation to input and all valid targets together
        if self.augment:
            # Collect all valid images for synchronized augmentation
            all_images = [input_img]
            valid_indices = []
            for i, img in enumerate(target_images):
                if img is not None:
                    all_images.append(img)
                    valid_indices.append(i)
            
            # Apply same augmentation to all
            all_images = self._apply_augmentation(all_images)
            
            # Unpack back
            input_img = all_images[0]
            for j, idx_t in enumerate(valid_indices):
                target_images[idx_t] = all_images[j + 1]

        # Convert to tensors
        input_tensor = TF.to_tensor(input_img)
        
        target_tensors = []
        for i, img in enumerate(target_images):
            if img is not None:
                target_tensors.append(TF.to_tensor(img))
            else:
                target_tensors.append(torch.zeros((1, 128, 128)))

        # Concatenate channels
        target_tensor = torch.cat(target_tensors, dim=0)
        mask_tensor = torch.cat(mask_tensors, dim=0)
        
        return input_tensor, target_tensor, mask_tensor

# Standard Transform (only used for resize now, augmentation handled separately)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ==========================================
# MODEL (U-Net - Larger Capacity)
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
# TRAINING LOOP
# ==========================================
def train():
    print("Loading Manifest...")
    try:
        dataset = CKMManifestDataset(
            MANIFEST_FILE, DATA_ROOT, 
            transform=transform, 
            augment=USE_AUGMENTATION
        )
        if USE_AUGMENTATION:
            print("Data augmentation ENABLED (flips + rotations)")
    except Exception as e:
        print(e)
        return

    # Use subset for faster testing
    if SUBSET_SIZE is not None and SUBSET_SIZE < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(SUBSET_SIZE))
        print(f"Using subset: {SUBSET_SIZE} samples")

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,       # Parallel data loading
        pin_memory=True,     # Faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    model = UNet(n_channels=1, n_classes=len(TARGET_COLS)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Use reduction='none' to apply mask element-wise
    criterion = nn.MSELoss(reduction='none')

    print(f"Starting Training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, targets, masks in pbar:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            masks = masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate raw loss
            raw_loss = criterion(outputs, targets)
            
            # Apply mask: Zero out loss for missing data
            masked_loss = raw_loss * masks
            
            # Normalize by number of valid elements (avoid div by zero)
            valid_elements = masks.sum()
            if valid_elements > 0:
                loss = masked_loss.sum() / valid_elements
            else:
                loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {epoch_loss/len(dataloader):.6f}")
        
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"ckm_epoch_with_augmentation_{epoch+1}.pth")

    print("Done.")

if __name__ == "__main__":
    train()