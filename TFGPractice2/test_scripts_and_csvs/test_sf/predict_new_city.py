import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your trained model file
MODEL_PATH = "../../CKMImagenet/ckm_epoch_with_augmentation_2.pth" 

# Path to the NEW city map you want to process
# (Change this to your new file)
INPUT_MAP_PATH = "C:\\Users\\guill\\OneDrive\\Documents\\TFG\\TFGPractice\\test\\San Fransisco\\San Fransisco.jpg" 

# Where to save the results
OUTPUT_DIR = "predicted_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The names of the 5 output channels (in the order trained)
CHANNEL_NAMES = ["Channel_Gain"]

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
# PREDICTION LOGIC
# ==========================================
def predict():
    # 1. Setup
    if not os.path.exists(INPUT_MAP_PATH):
        print(f"Error: Input map '{INPUT_MAP_PATH}' not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    model.eval() # Switch to evaluation mode (turns off BatchNorm/Dropout randomness)

    # 3. Process Input Image
    print("Processing input map...")
    input_img = Image.open(INPUT_MAP_PATH).convert('L') # Force grayscale
    
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # Normalizes to [0, 1]
    ])
    
    input_tensor = preprocess(input_img).unsqueeze(0).to(DEVICE) # Add batch dimension -> [1, 1, 128, 128]

    # 4. Run Inference
    with torch.no_grad(): # Disable gradient calculation for speed
        output_tensor = model(input_tensor)

    # 5. Save & Visualize Results
    print("Saving results...")
    
    # Remove batch dimension -> [5, 128, 128]
    output_tensor = output_tensor.squeeze(0).cpu() 
    
    plt.figure(figsize=(15, 8))
    
    # Plot Input
    plt.subplot(2, 3, 1)
    plt.title("Input City Map")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    # Plot Outputs
    for i, name in enumerate(CHANNEL_NAMES):
        channel_data = output_tensor[i].numpy()
        
        # Save as individual PNG
        save_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        plt.imsave(save_path, channel_data, cmap='inferno') # 'inferno' is good for heatmaps
        
        # Add to plot
        plt.subplot(2, 3, i+2)
        plt.title(f"Pred: {name}")
        plt.imshow(channel_data, cmap='inferno')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "full_comparison.png"))
    print(f"Done! Check the '{OUTPUT_DIR}' folder.")
    plt.show()

if __name__ == "__main__":
    predict()